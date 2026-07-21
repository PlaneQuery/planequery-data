"""Launch and monitor flight-algorithm jobs on AWS Batch.

Reads Batch metadata from ``PlanequeryFlightInferenceBatchStack`` outputs and
submits one AWS Batch job for the date range. Multi-day ranges use an array job
where each child runs one day using ``AWS_BATCH_JOB_ARRAY_INDEX``::

    python /app/src/data_engineering/flights/flights_from_algorithm.py \
        START_DATE END_DATE --ingest-to-iceberg


Usage::

    # Submit one durable job and monitor it:
    python scripts/run_flight_inference_batch_on_aws.py 2025-08-01 2025-08-10

    # Submit the job and exit; AWS Batch keeps the durable result, but the
    # local fail-fast monitor will not stop sibling array children:
    python scripts/run_flight_inference_batch_on_aws.py 2025-08-01 2025-08-10 --no-wait

    # Run multi-day array children sequentially:
    python scripts/run_flight_inference_batch_on_aws.py 2025-08-01 2025-08-10 --max-parallel 1

    # Allow AWS Batch to run more than one array child at a time:
    python scripts/run_flight_inference_batch_on_aws.py 2025-08-01 2025-08-10 --max-parallel 4

    # Use a larger container resource shape:
    python scripts/run_flight_inference_batch_on_aws.py 2025-08-01 2025-08-10 --size medium

For multi-day ranges, the array job parent is the durable "whole batch" result
in AWS Batch. While this launcher is polling, it fail-fast terminates the whole
array if any child fails after its Batch retries are exhausted. With
``--no-wait``, AWS Batch still marks the parent failed eventually, but remaining
children may continue until Batch naturally settles the array. A one-day range
is submitted as a regular Batch job.

Requires AWS credentials with ``batch:SubmitJob``, ``batch:DescribeJobs``,
``batch:TerminateJob``, ``batch:CancelJob``, and
``cloudformation:DescribeStacks``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from dataclasses import dataclass

import boto3


STACK_NAME = "PlanequeryFlightInferenceBatchStack"
DEFAULT_REGION = "us-east-2"
POLL_INTERVAL_S = 30
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED"}
QUEUE_STATUSES = {"SUBMITTED", "PENDING", "RUNNABLE"}
MAX_ARRAY_SIZE = 10_000
JOB_SIZES = ("small", "medium", "large")
JOB_DEFINITION_OUTPUTS = {
    "small": "JobDefinitionSmallArn",
    "medium": "JobDefinitionMediumArn",
    "large": "JobDefinitionLargeArn",
}


@dataclass(frozen=True)
class StackConfig:
    job_queue_arn: str
    job_definition_arns: dict[str, str]
    log_group: str
    compute_environment: str


def _load_stack_outputs(region: str) -> StackConfig:
    cfn = boto3.client("cloudformation", region_name=region)
    resp = cfn.describe_stacks(StackName=STACK_NAME)
    outputs = {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0]["Outputs"]}

    missing = [
        k
        for k in (
            "JobQueueArn",
            "LogGroupName",
            "ComputeEnvironmentName",
        )
        if k not in outputs
    ]
    if missing:
        raise RuntimeError(
            f"Stack {STACK_NAME} missing outputs: {missing}. "
            f"Re-deploy with `cdk deploy {STACK_NAME}`."
        )

    job_definition_arns = {
        size: outputs[output_key]
        for size, output_key in JOB_DEFINITION_OUTPUTS.items()
    }

    return StackConfig(
        job_queue_arn=outputs["JobQueueArn"],
        job_definition_arns=job_definition_arns,
        log_group=outputs["LogGroupName"],
        compute_environment=outputs["ComputeEnvironmentName"],
    )


def _validate_dates(start_date: str, end_date: str) -> tuple[dt.date, dt.date]:
    start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
    if end <= start:
        raise ValueError(f"end_date ({end}) must be > start_date ({start})")
    return start, end


def _batch_console_url(region: str, job_id: str) -> str:
    return f"https://{region}.console.aws.amazon.com/batch/home?region={region}#jobs/detail/{job_id}"


def _failed_child_count(aws_job: dict) -> int:
    summary = aws_job.get("arrayProperties", {}).get("statusSummary") or {}
    return int(summary.get("FAILED", 0))


def _stop_failed_array(
    batch_client,
    job_id: str,
    status: str,
    failed_children: int,
) -> None:
    reason = (
        "Fail-fast: stopping flight inference array after "
        f"{failed_children} child job(s) failed."
    )
    if status in QUEUE_STATUSES:
        batch_client.cancel_job(jobId=job_id, reason=reason)
    else:
        batch_client.terminate_job(jobId=job_id, reason=reason)


def _command_for_range(start: dt.date, end: dt.date) -> list[str]:
    command = [
        "python",
        "/app/src/data_engineering/flights/flights_from_algorithm.py",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    ]
    command.append("--write-to-s3")
    command.append("--ingest-to-iceberg")
    command.append("--prod")
    return command


def _submit_job(
    batch_client,
    cfg: StackConfig,
    start: dt.date,
    end: dt.date,
    max_parallel: int | None,
    job_size: str,
    run_ts: str,
) -> str:
    day_count = (end - start).days
    job_definition_arn = cfg.job_definition_arns[job_size]
    job_name = f"flights-{job_size}-{start:%Y-%m-%d}-{end:%Y-%m-%d}-{run_ts}"[:128]
    kwargs = {
        "jobName": job_name,
        "jobQueue": cfg.job_queue_arn,
        "jobDefinition": job_definition_arn,
        "containerOverrides": {"command": _command_for_range(start, end)},
    }
    if day_count > 1:
        kwargs["arrayProperties"] = {"size": day_count}
        if max_parallel == 1:
            kwargs["dependsOn"] = [{"type": "SEQUENTIAL"}]
    resp = batch_client.submit_job(**kwargs)
    job_id = resp["jobId"]
    job_type = "array job" if day_count > 1 else "job"
    print(f"Submitted {job_type} {start} .. {end - dt.timedelta(days=1)}: {job_id}")
    return job_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_date", help="Start date (inclusive), YYYY-MM-DD")
    parser.add_argument("end_date", help="End date (exclusive), YYYY-MM-DD")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--size",
        choices=JOB_SIZES,
        default="small",
        help=(
            "Batch job definition size. small=16 vCPU/60 GiB, "
            "medium=32 vCPU/120 GiB, large=64 vCPU/240 GiB."
        ),
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the array job and exit immediately without polling.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help=(
            "Array-child concurrency mode. Use 1 for AWS Batch SEQUENTIAL "
            "array dependencies; values >1 allow normal AWS Batch parallelism."
        ),
    )
    args = parser.parse_args()

    start, end = _validate_dates(args.start_date, args.end_date)
    day_count = (end - start).days
    if args.max_parallel is not None and args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")
    if day_count > MAX_ARRAY_SIZE:
        parser.error(
            f"AWS Batch array jobs support at most {MAX_ARRAY_SIZE} children; "
            f"date range has {day_count} days."
        )

    cfg = _load_stack_outputs(args.region)

    job_type = "array job" if day_count > 1 else "job"
    print(f"Scheduling {day_count} day(s) ({start} .. {end - dt.timedelta(days=1)}) as {job_type}")
    print(f"ComputeEnv:     {cfg.compute_environment}")
    print(f"JobQueue:       {cfg.job_queue_arn}")
    print(f"JobSize:        {args.size}")
    print(f"JobDefinition:  {cfg.job_definition_arns[args.size]}")
    print(f"LogGroup:       {cfg.log_group}")
    if day_count > 1 and args.max_parallel == 1:
        print("MaxParallel:    1 (AWS Batch SEQUENTIAL array dependency)")
    elif day_count > 1 and args.max_parallel is not None:
        print(f"MaxParallel:    {args.max_parallel} (AWS Batch-managed parallel array)")

    batch_client = boto3.client("batch", region_name=args.region)
    run_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")

    job_id = _submit_job(
        batch_client,
        cfg,
        start,
        end,
        args.max_parallel,
        args.size,
        run_ts,
    )
    print(f"Console:        {_batch_console_url(args.region, job_id)}")
    if args.no_wait:
        print(f"\n--no-wait: {job_type} will remain the durable batch result in AWS Batch.")
        return 0

    t0 = time.time()
    last_status = None
    last_summary = None
    termination_requested = False
    while True:
        desc = batch_client.describe_jobs(jobs=[job_id])
        jobs = desc.get("jobs", [])
        if not jobs:
            raise RuntimeError(f"Array job disappeared from DescribeJobs: {job_id}")
        aws_job = jobs[0]
        status = aws_job["status"]
        summary = aws_job.get("arrayProperties", {}).get("statusSummary")
        if status != last_status or summary != last_summary:
            elapsed_m = (time.time() - t0) / 60
            status_reason = aws_job.get("statusReason") or ""
            print(f"[{elapsed_m:6.1f}m] {job_type} {status} {status_reason}".rstrip())
            if summary:
                summary_parts = [
                    f"{state.lower()}={summary[state]}"
                    for state in sorted(summary)
                    if summary[state]
                ]
                print(f"           children: {', '.join(summary_parts)}")
            last_status = status
            last_summary = summary
        failed_children = _failed_child_count(aws_job)
        if day_count > 1 and failed_children and status not in TERMINAL_STATUSES:
            elapsed_m = (time.time() - t0) / 60
            if not termination_requested:
                print(
                    f"[{elapsed_m:6.1f}m] fail-fast: {failed_children} child "
                    f"job(s) failed; stopping array job"
                )
                _stop_failed_array(batch_client, job_id, status, failed_children)
                termination_requested = True
            else:
                print(
                    f"[{elapsed_m:6.1f}m] fail-fast termination requested; "
                    "waiting for AWS Batch to settle array job"
                )
        if status in TERMINAL_STATUSES:
            elapsed_m = (time.time() - t0) / 60
            print(f"\nCompleted in {elapsed_m:.1f} min: {job_type} {status}")
            return 0 if status == "SUCCEEDED" else 1
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
