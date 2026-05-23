"""ECS Fargate stack that runs the ADS-B historical ingest job.

The stack provisions:

* An ECR-backed Docker image, built locally from ``Dockerfile.adsb_insert`` and
  pushed to AWS during ``cdk deploy``.
* An ECS Fargate cluster on the default VPC.
* A task definition with enough CPU/memory/ephemeral storage to download a
  day of adsb.lol data, write Parquet, and append into the
  ``planequery_lake_dev.adsb_messages`` Iceberg table.
* IAM permissions for the Glue catalog and the
  ``planequery-dev-adsb-lake`` data bucket.

A companion CLI (``scripts/run_adsb_insert_on_aws.py``) reads the stack
outputs and invokes ``ecs:RunTask`` once per day, with a configurable
parallelism (default 3).
"""

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Stack,
    Token,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
)
from constructs import Construct


# Repo root, two levels up from this file: cdk/cdk/<this>.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Match the bucket created in PlanequeryDatalakeStack.
_DATA_BUCKET = "planequery-dev-lake"
_GLUE_DATABASE = "planequery_lake_dev"
_GLUE_TABLE = "adsb_messages"


class PlanequeryDatalakeAdsbInsertStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Networking: reuse the default VPC. Tasks run in public subnets with
        # ``assign_public_ip`` so they can reach raw.githubusercontent.com /
        # GitHub release assets without a NAT gateway.
        # ------------------------------------------------------------------
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)
        self.vpc = vpc

        security_group = ec2.SecurityGroup(
            self,
            "AdsbInsertSg",
            vpc=vpc,
            description="ADS-B Iceberg insert Fargate tasks (egress only)",
            allow_all_outbound=True,
        )
        self.security_group = security_group

        # ------------------------------------------------------------------
        # Build & push the docker image from the local machine to ECR.
        # ------------------------------------------------------------------
        image_asset = ecr_assets.DockerImageAsset(
            self,
            "AdsbInsertImage",
            directory=str(_REPO_ROOT),
            file="Dockerfile.adsb_insert",
            platform=ecr_assets.Platform.LINUX_ARM64,
            exclude=[
                "data/**",
                "notebooks/**",
                "tests/**",
                "cdk/cdk.out/**",
                ".venv/**",
                ".git/**",
                "**/__pycache__/**",
            ],
        )

        # ------------------------------------------------------------------
        # ECS cluster + log group
        # ------------------------------------------------------------------
        cluster = ecs.Cluster(
            self,
            "AdsbInsertCluster",
            vpc=vpc,
            cluster_name="planequery-dev-adsb-insert",
            container_insights=True,
        )
        self.cluster = cluster

        log_group = logs.LogGroup(
            self,
            "AdsbInsertLogs",
            log_group_name="/planequery/dev/adsb-insert",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        # ------------------------------------------------------------------
        # Task role: permissions the Python code uses at runtime.
        # ------------------------------------------------------------------
        task_role = iam.Role(
            self,
            "AdsbInsertTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Runtime role for ADS-B Iceberg insert tasks",
        )

        # S3: read/write the Iceberg data bucket.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListBucket",
                    "s3:ListBucketMultipartUploads",
                    "s3:GetBucketLocation",
                ],
                resources=[
                    f"arn:aws:s3:::{_DATA_BUCKET}",
                    f"arn:aws:s3:::{_DATA_BUCKET}/*",
                ],
            )
        )

        # S3: read/put access to the planequery-data bucket.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:ListBucket",
                    "s3:GetBucketLocation",
                ],
                resources=[
                    "arn:aws:s3:::planequery-dev-planequery-data",
                    "arn:aws:s3:::planequery-dev-planequery-data/*",
                ],
            )
        )

        # Daily pipeline handoff: the no-date ADS-B task writes the dates it
        # processed so the parent workflow can launch dependent feature jobs.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:PutParameter"],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/planequery/dev/daily-pipeline/*",
                ],
            )
        )

        # Glue catalog: PyIceberg's GlueCatalog reads/updates table metadata.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetDatabases",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:UpdateTable",
                    "glue:CreateTable",
                    "glue:GetTableVersion",
                    "glue:GetTableVersions",
                    "glue:GetPartition",
                    "glue:GetPartitions",
                    "glue:BatchGetPartition",
                    "glue:BatchCreatePartition",
                    "glue:BatchUpdatePartition",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{_GLUE_DATABASE}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{_GLUE_DATABASE}/{_GLUE_TABLE}",
                ],
            )
        )

        # ------------------------------------------------------------------
        # Task definition. Each task processes one or more days sequentially;
        # the CLI fans out to multiple tasks for day-level parallelism.
        # ------------------------------------------------------------------
        task_definition = ecs.FargateTaskDefinition(
            self,
            "AdsbInsertTaskDef",
            family="planequery-dev-adsb-insert",
            cpu=4096,            # 4 vCPU
            memory_limit_mib=16384,  # 10 GB
            ephemeral_storage_gib=100,
            task_role=task_role,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
        self.task_definition = task_definition

        container_definition = task_definition.add_container(
            "planequery-dev-adsb-insert",
            image=ecs.ContainerImage.from_docker_image_asset(image_asset),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="planequery-dev-adsb-insert",
                log_group=log_group,
            ),
            environment={
                "OUTPUT_DIR": "/app",
                "AWS_REGION": self.region,
            },
        )
        self.container_definition = container_definition

        # ------------------------------------------------------------------
        # Step Functions: durable serverless orchestration for multi-day
        # backfills.  One ECS Fargate task runs per date; MaxConcurrencyPath
        # lets the caller choose parallelism at execution-start time without
        # redeploying.
        # ------------------------------------------------------------------
        sfn_log_group = logs.LogGroup(
            self,
            "AdsbBackfillSfnLogs",
            log_group_name="/planequery/dev/adsb-backfill-sfn",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        sfn_role = iam.Role(
            self,
            "AdsbBackfillSfnRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Execution role for ADS-B backfill Step Functions state machine",
        )

        # Permissions required for Step Functions → CloudWatch Logs delivery.
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

        # Each Map iteration runs one Fargate task for the date string.
        # The container command is overridden per-iteration with the date.
        run_task_state = sfn_tasks.EcsRunTask(
            self,
            "RunAdsbInsertTask",
            cluster=cluster,
            task_definition=task_definition,
            launch_target=sfn_tasks.EcsFargateLaunchTarget(
                platform_version=ecs.FargatePlatformVersion.LATEST,
            ),
            assign_public_ip=True,
            subnets=ec2.SubnetSelection(subnets=vpc.public_subnets),
            security_groups=[security_group],
            container_overrides=[
                sfn_tasks.ContainerOverride(
                    container_definition=task_definition.default_container,
                    command=Token.as_list(
                        sfn.JsonPath.array(sfn.JsonPath.string_at("$"))
                    ),
                )
            ],
            propagated_tag_source=ecs.PropagatedTagSource.TASK_DEFINITION,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path=sfn.JsonPath.DISCARD,
        )

        # MaxConcurrencyPath lets the caller control parallelism at
        # execution-start time without modifying or redeploying the stack.
        map_state = sfn.Map(
            self,
            "BackfillDateMap",
            comment="Process each date as a Fargate task with bounded concurrency.",
            items_path=sfn.JsonPath.string_at("$.dates"),
            max_concurrency_path=sfn.JsonPath.string_at("$.max_concurrency"),
            result_path=sfn.JsonPath.DISCARD,
        )
        map_state.item_processor(run_task_state)

        state_machine = sfn.StateMachine(
            self,
            "AdsbBackfillStateMachine",
            state_machine_name="planequery-dev-adsb-backfill",
            definition_body=sfn.DefinitionBody.from_chainable(map_state),
            role=sfn_role,
            state_machine_type=sfn.StateMachineType.STANDARD,
            logs=sfn.LogOptions(
                destination=sfn_log_group,
                level=sfn.LogLevel.ERROR,
                include_execution_data=False,
            ),
        )

        # ------------------------------------------------------------------
        # Outputs consumed by scripts/run_adsb_insert_on_aws.py
        # ------------------------------------------------------------------
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "TaskDefinitionArn", value=task_definition.task_definition_arn)
        CfnOutput(self, "ContainerName", value="planequery-dev-adsb-insert")
        CfnOutput(self, "SecurityGroupId", value=security_group.security_group_id)
        CfnOutput(
            self,
            "SubnetIds",
            value=",".join(s.subnet_id for s in vpc.public_subnets),
        )
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
