"""Daily orchestration for PlaneQuery data processing.

The daily ADS-B stage runs the existing ADS-B insert image at 02:00 UTC with
no positional dates.  In that mode ``download_adsb_to_parquet.py`` checks
``get_adsblol_releases_to_process`` once and exits with code 42 when there is
no work.  The state machine catches that sentinel, waits 60 minutes, and starts
a fresh task up to six times. Once ADS-B succeeds, the workflow runs the
canonical flight algorithm for every newly available ADS-B day.
"""

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
)
from constructs import Construct

from cdk.planequery_datalake_adsb_insert_stack import PlanequeryDatalakeAdsbInsertStack
from cdk.planequery_flight_inference_batch_stack import PlanequeryFlightInferenceBatchStack


class PlanequeryDailyPipelineStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        adsb_insert_stack: PlanequeryDatalakeAdsbInsertStack,
        flight_inference_stack: PlanequeryFlightInferenceBatchStack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        manifest_parameter = sfn.Pass(
            self,
            "SetManifestParameter",
            parameters={
                "name.$": (
                    "States.Format('/planequery/dev/daily-pipeline/{}/adsb-manifest', "
                    "$$.Execution.Name)"
                ),
                "release_wait_count": 0,
            },
            result_path="$.manifest_parameter",
        )

        adsb_cluster_arn = (
            f"arn:{self.partition}:ecs:{self.region}:{self.account}:"
            "cluster/planequery-dev-adsb-insert"
        )
        adsb_task_definition_family = "planequery-dev-adsb-insert"
        adsb_container_name = "planequery-dev-adsb-insert"

        run_adsb_insert = sfn.CustomState(
            self,
            "RunAdsbInsertWithoutDates",
            state_json={
                "Type": "Task",
                "Resource": f"arn:{self.partition}:states:::ecs:runTask.sync",
                "Parameters": {
                    "Cluster": adsb_cluster_arn,
                    "TaskDefinition": adsb_task_definition_family,
                    "LaunchType": "FARGATE",
                    "PlatformVersion": "LATEST",
                    "PropagateTags": "TASK_DEFINITION",
                    "NetworkConfiguration": {
                        "AwsvpcConfiguration": {
                            "AssignPublicIp": "ENABLED",
                            "Subnets": [
                                subnet.subnet_id
                                for subnet in adsb_insert_stack.vpc.public_subnets
                            ],
                            "SecurityGroups": [
                                adsb_insert_stack.security_group.security_group_id
                            ],
                        },
                    },
                    "Overrides": {
                        "ContainerOverrides": [
                            {
                                "Name": adsb_container_name,
                                "Environment": [
                                    {
                                        "Name": "DAILY_PIPELINE_MANIFEST_SSM_PARAMETER",
                                        "Value.$": "$.manifest_parameter.name",
                                    },
                                ],
                            },
                        ],
                    },
                },
                "ResultSelector": {
                    "task_arn.$": "$.TaskArn",
                    "last_status.$": "$.LastStatus",
                    "stop_code.$": "$.StopCode",
                },
                "ResultPath": "$.adsb_insert_task",
            },
        )

        wait_for_release = sfn.Wait(
            self,
            "WaitSixtyMinutesForAdsbRelease",
            time=sfn.WaitTime.duration(Duration.minutes(60)),
        )
        increment_release_wait_count = sfn.Pass(
            self,
            "IncrementAdsbReleaseWaitCount",
            parameters={
                "name.$": "$.manifest_parameter.name",
                "release_wait_count.$": (
                    "States.MathAdd($.manifest_parameter.release_wait_count, 1)"
                ),
            },
            result_path="$.manifest_parameter",
        )
        parse_adsb_failure = sfn.Pass(
            self,
            "ParseAdsbInsertFailure",
            parameters={
                "cause.$": "States.StringToJson($.adsb_failure.Cause)",
            },
            result_path="$.adsb_failure_detail",
        )
        fail_adsb_insert = sfn.Fail(
            self,
            "AdsbInsertFailed",
            cause="ADS-B insert task failed with a non-retryable exit code.",
        )
        fail_adsb_release_unavailable = sfn.Fail(
            self,
            "AdsbReleaseUnavailableAfterSixWaits",
            cause="ADS-B release was unavailable after six hourly waits.",
        )
        no_release_exit_code = sfn.Condition.or_(
            self._exit_code_equals("$.adsb_failure_detail.cause.Containers[0].ExitCode", 42),
            self._exit_code_equals("$.adsb_failure_detail.cause.Tasks[0].Containers[0].ExitCode", 42),
            self._exit_code_equals("$.adsb_failure_detail.cause.containers[0].exitCode", 42),
            self._exit_code_equals("$.adsb_failure_detail.cause.tasks[0].containers[0].exitCode", 42),
        )
        no_release_with_waits_remaining = sfn.Condition.and_(
            no_release_exit_code,
            sfn.Condition.number_less_than(
                "$.manifest_parameter.release_wait_count",
                6,
            ),
        )
        handle_adsb_failure = sfn.Choice(self, "WasAdsbReleaseUnavailable")
        handle_adsb_failure.when(
            no_release_with_waits_remaining,
            increment_release_wait_count.next(wait_for_release).next(run_adsb_insert),
        )
        handle_adsb_failure.when(no_release_exit_code, fail_adsb_release_unavailable)
        handle_adsb_failure.otherwise(fail_adsb_insert)
        run_adsb_insert.add_catch(
            parse_adsb_failure.next(handle_adsb_failure),
            errors=["States.ALL"],
            result_path="$.adsb_failure",
        )

        adsb_processed = sfn.Pass(
            self,
            "AdsbReleaseProcessed",
            comment="Future daily pipeline stages can be chained after this state.",
        )
        get_adsb_manifest = sfn_tasks.CallAwsService(
            self,
            "ReadAdsbProcessedDatesManifest",
            service="ssm",
            action="getParameter",
            parameters={
                "Name": sfn.JsonPath.string_at("$.manifest_parameter.name"),
            },
            iam_resources=[
                f"arn:aws:ssm:{self.region}:{self.account}:parameter/planequery/dev/daily-pipeline/*",
            ],
            result_selector={
                "manifest.$": "States.StringToJson($.Parameter.Value)",
            },
            result_path="$.adsb_manifest",
        )
        run_flight_inference_job = sfn_tasks.BatchSubmitJob(
            self,
            "RunFlightInferenceForDay",
            job_definition_arn=flight_inference_stack.job_definition_name,
            job_name=sfn.JsonPath.string_at("$.job_name"),
            job_queue_arn=flight_inference_stack.job_queue_arn,
            container_overrides=sfn_tasks.BatchContainerOverrides(
                command=sfn.JsonPath.list_at("$.command"),
            ),
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path=sfn.JsonPath.DISCARD,
        )
        run_flight_inference = sfn.Map(
            self,
            "RunFlightAlgorithmForAdsbDays",
            comment="Run the flight algorithm once for each newly available ADS-B day.",
            items_path=sfn.JsonPath.string_at("$.adsb_manifest.manifest.flight_inference_jobs"),
            max_concurrency=1,
            result_path=sfn.JsonPath.DISCARD,
        )
        run_flight_inference.item_processor(run_flight_inference_job)
        done = sfn.Succeed(self, "DailyPipelineComplete")
        definition = (
            manifest_parameter
            .next(run_adsb_insert)
            .next(adsb_processed)
            .next(get_adsb_manifest)
            .next(run_flight_inference)
            .next(done)
        )

        sfn_log_group = logs.LogGroup.from_log_group_name(
            self,
            "DailyPipelineSfnLogs",
            log_group_name="/planequery/dev/daily-pipeline-sfn",
        )

        sfn_role = iam.Role(
            self,
            "DailyPipelineSfnRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Execution role for the PlaneQuery daily pipeline",
        )
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogDelivery",
                    "logs:GetLogDelivery",
                    "logs:UpdateLogDelivery",
                    "logs:DeleteLogDelivery",
                    "logs:ListLogDeliveries",
                    "logs:PutResourcePolicy",
                    "logs:DescribeResourcePolicies",
                    "logs:DescribeLogGroups",
                ],
                resources=["*"],
            )
        )
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:RunTask"],
                resources=[
                    (
                        f"arn:{self.partition}:ecs:{self.region}:{self.account}:"
                        f"task-definition/{adsb_task_definition_family}:*"
                    ),
                ],
            )
        )
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:DescribeTasks", "ecs:StopTask"],
                resources=["*"],
            )
        )
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[
                    (
                        f"arn:{self.partition}:iam::{self.account}:"
                        "role/PlanequeryDatalakeAdsbIns-AdsbInsertTask*"
                    ),
                ],
            )
        )
        sfn_role.add_to_policy(
            iam.PolicyStatement(
                actions=["events:DescribeRule", "events:PutRule", "events:PutTargets"],
                resources=[
                    (
                        f"arn:{self.partition}:events:{self.region}:{self.account}:"
                        "rule/StepFunctionsGetEventsForECSTaskRule"
                    ),
                ],
            )
        )

        state_machine = sfn.StateMachine(
            self,
            "DailyPipelineStateMachine",
            state_machine_name="planequery-dev-daily-pipeline",
            definition_body=sfn.DefinitionBody.from_chainable(definition),
            role=sfn_role,
            state_machine_type=sfn.StateMachineType.STANDARD,
            logs=sfn.LogOptions(
                destination=sfn_log_group,
                level=sfn.LogLevel.ERROR,
                include_execution_data=False,
            ),
        )

        schedule = events.Rule(
            self,
            "DailyPipelineSchedule",
            rule_name="planequery-dev-daily-pipeline",
            schedule=events.Schedule.cron(minute="0", hour="2"),
            description="Start the PlaneQuery daily pipeline at 02:00 UTC.",
        )
        schedule.add_target(targets.SfnStateMachine(state_machine))

        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
        CfnOutput(self, "ScheduleRuleName", value=schedule.rule_name)

    def _exit_code_equals(self, path: str, exit_code: int) -> sfn.Condition:
        return sfn.Condition.and_(
            sfn.Condition.is_present(path),
            sfn.Condition.number_equals(path, exit_code),
        )
