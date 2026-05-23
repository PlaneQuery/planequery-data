"""Trigger and monitor the PlaneQuery daily pipeline on AWS.

Reads ``StateMachineArn`` from ``PlanequeryDailyPipelineStack`` outputs, starts
one Step Functions Standard Workflow execution, and optionally polls until it
finishes.

The daily schedule normally starts the state machine at 02:00 UTC with empty
input. This script does the same thing for a manual run.

Usage::

    # Start and wait for completion (prints status every 30 s):
    python scripts/run_daily_pipeline_on_aws.py

    # Fire-and-forget; monitor in the AWS console:
    python scripts/run_daily_pipeline_on_aws.py --no-wait

    # Monitor an execution that was already started:
    python scripts/run_daily_pipeline_on_aws.py --execution-arn arn:aws:states:...

Requires AWS credentials with ``states:StartExecution``,
``states:DescribeExecution``, ``states:GetExecutionHistory``, and
``cloudformation:DescribeStacks``. ``--tail-logs`` also requires
``logs:DescribeLogStreams`` and ``logs:GetLogEvents``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import dataclass, field

import boto3


STACK_NAME = "PlanequeryDailyPipelineStack"
ADSB_STACK_NAME = "PlanequeryDatalakeAdsbInsertStack"
INFERENCE_FEATURES_STACK_NAME = "PlanequeryInferenceFeaturesStack"
FLIGHT_INFERENCE_STACK_NAME = "PlanequeryFlightInferenceBatchStack"
DEFAULT_REGION = "us-east-2"
POLL_INTERVAL_S = 30
TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"}
ECS_TASK_ARN_RE = re.compile(r"arn:aws[a-zA-Z-]*:ecs:[^:\s]+:\d+:task/[^\s\"']+")
SFN_EXECUTION_ARN_RE = re.compile(
    r"arn:aws[a-zA-Z-]*:states:[^:\s]+:\d+:execution:[^\s\"']+"
)
FLIGHT_INFERENCE_LOG_PREFIXES = (
    "planequery-dev-flight-inference-small",
    "planequery-dev-flight-inference-medium",
    "planequery-dev-flight-inference-large",
)


@dataclass(frozen=True)
class StackConfig:
    state_machine_arn: str
    schedule_rule_name: str
    adsb_log_group: str | None = None
    inference_features_log_group: str | None = None
    flight_inference_log_group: str | None = None


@dataclass(frozen=True)
class ResourceRefs:
    ecs_task_arns: set[str] = field(default_factory=set)
    sfn_execution_arns: set[str] = field(default_factory=set)
    batch_job_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class LogSource:
    label: str
    log_group: str
    log_stream: str


def _load_stack_outputs(region: str, *, include_log_groups: bool = False) -> StackConfig:
    cfn = boto3.client("cloudformation", region_name=region)
    outputs = _describe_stack_outputs(cfn, STACK_NAME)

    missing = [
        k
        for k in (
            "StateMachineArn",
            "ScheduleRuleName",
        )
        if k not in outputs
    ]
    if missing:
        raise RuntimeError(
            f"Stack {STACK_NAME} missing outputs: {missing}. "
            f"Re-deploy with `cdk deploy {STACK_NAME}`."
        )

    if not include_log_groups:
        return StackConfig(
            state_machine_arn=outputs["StateMachineArn"],
            schedule_rule_name=outputs["ScheduleRuleName"],
        )

    return StackConfig(
        state_machine_arn=outputs["StateMachineArn"],
        schedule_rule_name=outputs["ScheduleRuleName"],
        adsb_log_group=_stack_output(cfn, ADSB_STACK_NAME, "LogGroupName"),
        inference_features_log_group=_stack_output(
            cfn,
            INFERENCE_FEATURES_STACK_NAME,
            "LogGroupName",
        ),
        flight_inference_log_group=_stack_output(
            cfn,
            FLIGHT_INFERENCE_STACK_NAME,
            "LogGroupName",
        ),
    )


def _describe_stack_outputs(cfn, stack_name: str) -> dict[str, str]:
    resp = cfn.describe_stacks(StackName=stack_name)
    return {o["OutputKey"]: o["OutputValue"] for o in resp["Stacks"][0].get("Outputs", [])}


def _stack_output(cfn, stack_name: str, output_key: str) -> str | None:
    try:
        return _describe_stack_outputs(cfn, stack_name).get(output_key)
    except Exception as exc:  # noqa: BLE001 - keep log tailing optional.
        print(f"Warning: could not read {stack_name}.{output_key}: {exc}")
        return None


def _console_url(region: str, execution_arn: str) -> str:
    encoded = execution_arn.replace(":", "%3A").replace("/", "%2F")
    return (
        f"https://{region}.console.aws.amazon.com/states/home"
        f"?region={region}#/executions/details/{encoded}"
    )


def _latest_state_name(sfn, execution_arn: str) -> str | None:
    resp = sfn.get_execution_history(
        executionArn=execution_arn,
        maxResults=25,
        reverseOrder=True,
    )
    for event in resp.get("events", []):
        details = event.get("stateEnteredEventDetails")
        if details:
            return details.get("name")
    return None


def _failure_line(desc: dict) -> str | None:
    error = desc.get("error")
    cause = desc.get("cause")
    if not error and not cause:
        return None
    if cause and len(cause) > 500:
        cause = f"{cause[:500]}..."
    if error and cause:
        return f"{error}: {cause}"
    return error or cause


def _iter_execution_events(sfn, execution_arn: str):
    kwargs = {
        "executionArn": execution_arn,
        "maxResults": 1000,
        "reverseOrder": False,
    }
    while True:
        resp = sfn.get_execution_history(**kwargs)
        yield from resp.get("events", [])
        token = resp.get("nextToken")
        if not token:
            break
        kwargs["nextToken"] = token


def _extract_resource_refs(value) -> ResourceRefs:
    ecs_task_arns: set[str] = set()
    sfn_execution_arns: set[str] = set()
    batch_job_ids: set[str] = set()

    def visit(item) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                key_lower = key.lower()
                if isinstance(nested, str):
                    if key_lower in {"jobid", "job_id"}:
                        batch_job_ids.add(nested)
                    if "executionarn" in key_lower:
                        sfn_execution_arns.add(nested)
                    if "taskarn" in key_lower:
                        ecs_task_arns.add(nested)
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            ecs_task_arns.update(ECS_TASK_ARN_RE.findall(item))
            sfn_execution_arns.update(SFN_EXECUTION_ARN_RE.findall(item))
            stripped = item.strip()
            if stripped and stripped[0] in "[{":
                try:
                    visit(json.loads(stripped))
                except json.JSONDecodeError:
                    pass

    visit(value)
    return ResourceRefs(
        ecs_task_arns=ecs_task_arns,
        sfn_execution_arns=sfn_execution_arns,
        batch_job_ids=batch_job_ids,
    )


def _merge_resource_refs(refs: ResourceRefs, other: ResourceRefs) -> ResourceRefs:
    return ResourceRefs(
        ecs_task_arns=refs.ecs_task_arns | other.ecs_task_arns,
        sfn_execution_arns=refs.sfn_execution_arns | other.sfn_execution_arns,
        batch_job_ids=refs.batch_job_ids | other.batch_job_ids,
    )


def _discover_execution_resources(
    sfn,
    execution_arn: str,
    *,
    include_nested: bool = True,
) -> ResourceRefs:
    refs = ResourceRefs()
    seen_executions = {execution_arn}
    executions_to_scan = [execution_arn]

    while executions_to_scan:
        current_arn = executions_to_scan.pop()
        for event in _iter_execution_events(sfn, current_arn):
            event_refs = _extract_resource_refs(event)
            refs = _merge_resource_refs(refs, event_refs)
            if include_nested:
                for nested_arn in sorted(event_refs.sfn_execution_arns):
                    if nested_arn not in seen_executions:
                        seen_executions.add(nested_arn)
                        executions_to_scan.append(nested_arn)

    return ResourceRefs(
        ecs_task_arns=refs.ecs_task_arns,
        sfn_execution_arns=refs.sfn_execution_arns - {execution_arn},
        batch_job_ids=refs.batch_job_ids,
    )


def _task_id_from_arn(task_arn: str) -> str:
    return task_arn.rsplit("/", 1)[-1]


def _task_log_source(config: StackConfig, task_arn: str) -> LogSource | None:
    task_id = _task_id_from_arn(task_arn)
    if "planequery-dev-adsb-insert" in task_arn:
        if not config.adsb_log_group:
            return None
        return LogSource(
            label=f"adsb:{task_id[:8]}",
            log_group=config.adsb_log_group,
            log_stream=f"planequery-dev-adsb-insert/planequery-dev-adsb-insert/{task_id}",
        )
    if "planequery-dev-inference-features" in task_arn:
        if not config.inference_features_log_group:
            return None
        return LogSource(
            label=f"features:{task_id[:8]}",
            log_group=config.inference_features_log_group,
            log_stream=(
                "planequery-dev-inference-features/"
                f"planequery-dev-inference-features/{task_id}"
            ),
        )
    return None


def _batch_log_stream_prefixes(job_id: str) -> list[str]:
    return [f"{prefix}/default/{job_id}" for prefix in FLIGHT_INFERENCE_LOG_PREFIXES]


def _find_log_streams(logs, log_group: str, prefixes: list[str]) -> list[str]:
    streams: list[str] = []
    for prefix in prefixes:
        resp = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=prefix,
            limit=5,
        )
        streams.extend(s["logStreamName"] for s in resp.get("logStreams", []))
    return streams


class DailyPipelineLogTailer:
    def __init__(
        self,
        *,
        sfn,
        logs,
        config: StackConfig,
        execution_arn: str,
        start_time: dt.datetime,
        max_events_per_poll: int,
    ) -> None:
        self.sfn = sfn
        self.logs = logs
        self.config = config
        self.execution_arn = execution_arn
        self.start_time_ms = int(start_time.timestamp() * 1000)
        self.max_events_per_poll = max_events_per_poll
        self.sources: dict[tuple[str, str], LogSource] = {}
        self.next_tokens: dict[tuple[str, str], str] = {}
        self.batch_sources_seen: set[str] = set()
        self.known_tasks: set[str] = set()

    def discover_sources(self) -> None:
        refs = _discover_execution_resources(self.sfn, self.execution_arn)
        for task_arn in sorted(refs.ecs_task_arns - self.known_tasks):
            source = _task_log_source(self.config, task_arn)
            if not source:
                self.known_tasks.add(task_arn)
                continue
            streams = _find_log_streams(self.logs, source.log_group, [source.log_stream])
            if not streams:
                continue
            self.known_tasks.add(task_arn)
            for stream in streams:
                discovered_source = LogSource(
                    label=source.label,
                    log_group=source.log_group,
                    log_stream=stream,
                )
                self.sources[
                    (discovered_source.log_group, discovered_source.log_stream)
                ] = discovered_source

        if not self.config.flight_inference_log_group:
            return
        for job_id in sorted(refs.batch_job_ids - self.batch_sources_seen):
            streams = _find_log_streams(
                self.logs,
                self.config.flight_inference_log_group,
                _batch_log_stream_prefixes(job_id),
            )
            if not streams:
                continue
            self.batch_sources_seen.add(job_id)
            for stream in streams:
                source = LogSource(
                    label=f"flights:{job_id[:8]}",
                    log_group=self.config.flight_inference_log_group,
                    log_stream=stream,
                )
                self.sources[(source.log_group, source.log_stream)] = source

    def print_new_events(self) -> None:
        printed = 0
        for key, source in sorted(self.sources.items(), key=lambda item: item[1].label):
            if printed >= self.max_events_per_poll:
                return
            kwargs = {
                "logGroupName": source.log_group,
                "logStreamName": source.log_stream,
                "startFromHead": True,
            }
            token = self.next_tokens.get(key)
            if token:
                kwargs["nextToken"] = token
            else:
                kwargs["startTime"] = self.start_time_ms
            resp = self.logs.get_log_events(**kwargs)
            self.next_tokens[key] = resp["nextForwardToken"]
            for event in resp.get("events", []):
                if printed >= self.max_events_per_poll:
                    return
                ts = dt.datetime.fromtimestamp(
                    event["timestamp"] / 1000,
                    tz=dt.timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ")
                message = event.get("message", "").rstrip()
                if message:
                    print(f"[log {source.label} {ts}] {message}")
                    printed += 1


def _start_execution(sfn, state_machine_arn: str, execution_name: str | None) -> str:
    if execution_name is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
        execution_name = f"daily-manual-{ts}"
    resp = sfn.start_execution(
        stateMachineArn=state_machine_arn,
        name=execution_name[:80],
        input=json.dumps({}),
    )
    return resp["executionArn"]


def _monitor_execution(
    sfn,
    region: str,
    execution_arn: str,
    poll_interval_s: int,
    *,
    log_tailer: DailyPipelineLogTailer | None = None,
) -> int:
    console_url = _console_url(region, execution_arn)
    print(f"Console:           {console_url}\n")

    t0 = time.time()
    last_line: str | None = None
    while True:
        desc = sfn.describe_execution(executionArn=execution_arn)
        status = desc["status"]
        elapsed_m = (time.time() - t0) / 60
        state_name = _latest_state_name(sfn, execution_arn)
        line = f"[{elapsed_m:6.1f}m] {status}"
        if state_name:
            line = f"{line} ({state_name})"
        if line != last_line or status in TERMINAL_STATUSES:
            print(line)
            last_line = line
        if log_tailer:
            log_tailer.discover_sources()
            log_tailer.print_new_events()
        if status in TERMINAL_STATUSES:
            break
        time.sleep(poll_interval_s)

    elapsed_m = (time.time() - t0) / 60
    print(f"\nExecution {status} in {elapsed_m:.1f} min")
    failure = _failure_line(desc)
    if failure:
        print(f"Failure: {failure}")
    if status != "SUCCEEDED":
        print(f"Inspect execution details: {console_url}")

    return 0 if status == "SUCCEEDED" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default: {DEFAULT_REGION})",
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional Step Functions execution name. Defaults to "
            "`daily-manual-<UTC timestamp>`."
        ),
    )
    parser.add_argument(
        "--execution-arn",
        default=None,
        help="Monitor an existing daily pipeline execution instead of starting one.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start the execution and exit immediately without polling.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=POLL_INTERVAL_S,
        help=f"Polling interval in seconds (default: {POLL_INTERVAL_S}).",
    )
    parser.add_argument(
        "--tail-logs",
        action="store_true",
        help="Tail CloudWatch logs for ECS and Batch work launched by the pipeline.",
    )
    parser.add_argument(
        "--log-lines-per-poll",
        type=int,
        default=100,
        help="Maximum CloudWatch log lines to print per poll when using --tail-logs.",
    )
    args = parser.parse_args()

    if args.poll_interval < 1:
        parser.error("--poll-interval must be >= 1")
    if args.log_lines_per_poll < 1:
        parser.error("--log-lines-per-poll must be >= 1")
    if args.execution_arn and args.name:
        parser.error("--name can only be used when starting a new execution")
    if args.execution_arn and args.no_wait:
        parser.error("--no-wait can only be used when starting a new execution")

    sfn = boto3.client("stepfunctions", region_name=args.region)

    if args.execution_arn:
        execution_arn = args.execution_arn
        print(f"Monitoring execution: {execution_arn}")
        desc = sfn.describe_execution(executionArn=execution_arn)
        log_tailer = None
        if args.tail_logs:
            cfg = _load_stack_outputs(args.region, include_log_groups=True)
            logs = boto3.client("logs", region_name=args.region)
            log_tailer = DailyPipelineLogTailer(
                sfn=sfn,
                logs=logs,
                config=cfg,
                execution_arn=execution_arn,
                start_time=desc["startDate"],
                max_events_per_poll=args.log_lines_per_poll,
            )
        return _monitor_execution(
            sfn,
            args.region,
            execution_arn,
            args.poll_interval,
            log_tailer=log_tailer,
        )

    cfg = _load_stack_outputs(args.region, include_log_groups=args.tail_logs)
    print("Starting daily pipeline")
    print(f"StateMachine:      {cfg.state_machine_arn}")
    print(f"ScheduleRule:      {cfg.schedule_rule_name}")
    print("Input:             {}")

    execution_arn = _start_execution(sfn, cfg.state_machine_arn, args.name)
    print(f"\nExecution started: {execution_arn}")

    if args.no_wait:
        print(f"Console:           {_console_url(args.region, execution_arn)}")
        print("\n--no-wait: exiting. Monitor progress in the AWS console.")
        return 0

    log_tailer = None
    if args.tail_logs:
        logs = boto3.client("logs", region_name=args.region)
        log_tailer = DailyPipelineLogTailer(
            sfn=sfn,
            logs=logs,
            config=cfg,
            execution_arn=execution_arn,
            start_time=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=2),
            max_events_per_poll=args.log_lines_per_poll,
        )

    return _monitor_execution(
        sfn,
        args.region,
        execution_arn,
        args.poll_interval,
        log_tailer=log_tailer,
    )


if __name__ == "__main__":
    sys.exit(main())
