"""Launch ADS-B Iceberg insert jobs via AWS Step Functions.

Reads ``StateMachineArn`` (and other metadata) from
``PlanequeryDatalakeAdsbInsertStack`` outputs, then starts a single Step
Functions Standard Workflow execution that fans out one ECS Fargate task per
day in ``[start_date, end_date)``.

Concurrency is controlled at execution time via ``--max-parallel``; no
redeployment is required to change it.  The state machine automatically
retries each failed day up to 2 times before marking it as an error.

Because execution state lives entirely in AWS, you can run ``--no-wait`` and
the backfill will continue even after this process exits.

Usage::

    # Start and wait for completion (prints status every 30 s):
    python scripts/run_adsb_insert_on_aws.py 2025-01-01 2026-01-01

    # Fire-and-forget; monitor in the AWS console:
    python scripts/run_adsb_insert_on_aws.py 2025-01-01 2026-01-01 --no-wait

    # Tune concurrency (ECS soft limit is 100 concurrent tasks per cluster):
    python scripts/run_adsb_insert_on_aws.py 2025-01-01 2026-01-01 --max-parallel 10

Requires AWS credentials with ``states:StartExecution``,
``states:DescribeExecution``, and ``cloudformation:DescribeStacks``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass

import boto3

STACK_NAME = "PlanequeryDatalakeAdsbInsertStack"
DEFAULT_REGION = "us-east-2"
DEFAULT_CONCURRENCY = 3
POLL_INTERVAL_S = 30


@dataclass(frozen=True)
class StackConfig:
    cluster: str
    task_definition: str
    container_name: str
    subnets: list[str]
    security_group: str
    log_group: str
    state_machine_arn: str


def _load_stack_outputs(region: str) -> StackConfig:
    cfn = boto3.client("cloudformation", region_name=region)
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}

    missing = [
        k for k in (
            "ClusterName",
            "TaskDefinitionArn",
            "ContainerName",
            "SubnetIds",
            "SecurityGroupId",
            "LogGroupName",
            "StateMachineArn",
        ) if k not in outputs
    ]
    if missing:
        raise RuntimeError(
            f"Stack {STACK_NAME} missing outputs: {missing}. "
            "Re-deploy with `cdk deploy PlanequeryDatalakeAdsbInsertStack`."
        )

    return StackConfig(
        cluster=outputs["ClusterName"],
        task_definition=outputs["TaskDefinitionArn"],
        container_name=outputs["ContainerName"],
        subnets=[s for s in outputs["SubnetIds"].split(",") if s],
        security_group=outputs["SecurityGroupId"],
        log_group=outputs["LogGroupName"],
        state_machine_arn=outputs["StateMachineArn"],
    )


def _daterange(start: dt.date, end: dt.date) -> list[str]:
    if end <= start:
        raise ValueError(f"end_date ({end}) must be > start_date ({start})")
    out: list[str] = []
    cur = start
    while cur < end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += dt.timedelta(days=1)
    return out


def _print_memory_stats(
    region: str,
    cluster: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
) -> None:
    """Query ECS/ContainerInsights and print peak + average RAM across all tasks."""
    cw = boto3.client("cloudwatch", region_name=region)
    resp = cw.get_metric_statistics(
        Namespace="ECS/ContainerInsights",
        MetricName="MemoryUtilized",
        Dimensions=[{"Name": "ClusterName", "Value": cluster}],
        StartTime=start_time,
        EndTime=end_time,
        Period=60,
        Statistics=["Maximum", "Average"],
    )
    datapoints = resp.get("Datapoints", [])
    if not datapoints:
        print("\nNo memory utilization datapoints found in CloudWatch.")
        return
    max_mb = max(d["Maximum"] for d in datapoints)
    avg_mb = sum(d["Average"] for d in datapoints) / len(datapoints)
    print(f"\nContainer memory utilization (cluster={cluster}):")
    print(f"  Max:  {max_mb:,.0f} MB  ({max_mb / 1024:.2f} GB)")
    print(f"  Avg:  {avg_mb:,.0f} MB  ({avg_mb / 1024:.2f} GB)")


def _console_url(region: str, execution_arn: str) -> str:
    encoded = execution_arn.replace(":", "%3A").replace("/", "%2F")
    return (
        f"https://{region}.console.aws.amazon.com/states/home"
        f"?region={region}#/executions/details/{encoded}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_date", help="Start date (inclusive), YYYY-MM-DD")
    parser.add_argument("end_date", help="End date (exclusive), YYYY-MM-DD")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=(
            f"Max concurrent Fargate tasks (default: {DEFAULT_CONCURRENCY}). "
            "ECS soft limit is 100 concurrent tasks per cluster."
        ),
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the execution and exit immediately without polling.",
    )
    args = parser.parse_args()

    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    dates = _daterange(start, end)

    print(
        f"Scheduling {len(dates)} day(s) "
        f"({start} .. {end - dt.timedelta(days=1)}) "
        f"with max_concurrency={args.max_parallel}"
    )

    cfg = _load_stack_outputs(args.region)
    print(f"StateMachine:   {cfg.state_machine_arn}")
    print(f"Cluster:        {cfg.cluster}")
    print(f"LogGroup:       {cfg.log_group}")

    sfn = boto3.client("stepfunctions", region_name=args.region)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    execution_name = (
        f"adsb-{args.start_date}-{args.end_date}-c{args.max_parallel}-{ts}"
    )[:80]

    resp = sfn.start_execution(
        stateMachineArn=cfg.state_machine_arn,
        name=execution_name,
        input=json.dumps({
            "dates": dates,
            "max_concurrency": args.max_parallel,
        }),
    )
    execution_arn = resp["executionArn"]
    console_url = _console_url(args.region, execution_arn)

    print(f"\nExecution started: {execution_arn}")
    print(f"Console:           {console_url}\n")

    if args.no_wait:
        print("--no-wait: exiting. Monitor progress in the AWS console.")
        return 0

    # Poll until the execution reaches a terminal state.
    t0 = time.time()
    exec_start_time = dt.datetime.now(dt.timezone.utc)
    while True:
        desc = sfn.describe_execution(executionArn=execution_arn)
        status = desc["status"]
        elapsed_m = (time.time() - t0) / 60
        print(f"[{elapsed_m:6.1f}m] {status}")
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"):
            break
        time.sleep(POLL_INTERVAL_S)

    exec_end_time = dt.datetime.now(dt.timezone.utc)
    elapsed_m = (time.time() - t0) / 60
    print(f"\nExecution {status} in {elapsed_m:.1f} min")
    if status != "SUCCEEDED":
        print(f"Inspect failed iterations: {console_url}")

    _print_memory_stats(args.region, cfg.cluster, exec_start_time, exec_end_time)

    return 0 if status == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
