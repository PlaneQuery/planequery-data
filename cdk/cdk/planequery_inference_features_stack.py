"""ECS Fargate stack that runs the inference-feature build job.

The stack provisions:

* An ECR-backed Docker image, built locally from
  ``Dockerfile.inference_features`` and pushed to AWS during ``cdk deploy``.
* An ECS Fargate cluster on the default VPC.
  day of ADS-B inference feature computation.
* IAM permissions for Athena, Glue, the ADS-B data lake bucket, the
  ``planequery-dev-data`` features bucket, and Athena results.
* A Step Functions state machine that fans out one Fargate task per day
  with bounded concurrency selected at execution-start time.

A companion CLI (``scripts/run_inference_features_on_aws.py``) reads the
stack outputs and starts an execution.
"""

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as sfn_tasks,
)
from constructs import Construct


_REPO_ROOT = Path(__file__).resolve().parents[2]

_ADSB_LAKE_BUCKET = "planequery-dev-adsb-lake"
_FEATURES_BUCKET = "planequery-dev-planequery-data"
_ATHENA_RESULTS_BUCKET = "planequery-dev-athena-results"
_GLUE_DATABASE = "planequery_lake_dev"
_GLUE_TABLE = "adsb_messages"
_ATHENA_WORKGROUP = "planequery-workgroup-dev"


class PlanequeryInferenceFeaturesStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Networking: reuse default VPC; tasks run in public subnets so they
        # can reach Athena / S3 without a NAT gateway.
        # ------------------------------------------------------------------
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        security_group = ec2.SecurityGroup(
            self,
            "InferenceFeaturesSg",
            vpc=vpc,
            description="Inference-feature Fargate tasks (egress only)",
            allow_all_outbound=True,
        )

        # ------------------------------------------------------------------
        # Build & push the docker image from the local machine to ECR.
        # ------------------------------------------------------------------
        image_asset = ecr_assets.DockerImageAsset(
            self,
            "InferenceFeaturesImage",
            directory=str(_REPO_ROOT),
            file="Dockerfile.inference_features",
            platform=ecr_assets.Platform.LINUX_ARM64,
            exclude=[
                "data/raw/**",
                "data/model/**",
                "data/features/**",
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
            "InferenceFeaturesCluster",
            vpc=vpc,
            cluster_name="planequery-dev-inference-features",
            container_insights=True,
        )

        log_group = logs.LogGroup(
            self,
            "InferenceFeaturesLogs",
            log_group_name="/planequery/dev/inference-features",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        # ------------------------------------------------------------------
        # Task role: permissions the Python code uses at runtime.
        # ------------------------------------------------------------------
        task_role = iam.Role(
            self,
            "InferenceFeaturesTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Runtime role for inference-feature build tasks",
        )

        # S3: read the Iceberg data bucket; write features to the data bucket;
        # use the Athena results bucket for query output staging.
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
                    f"arn:aws:s3:::{_ADSB_LAKE_BUCKET}",
                    f"arn:aws:s3:::{_ADSB_LAKE_BUCKET}/*",
                    f"arn:aws:s3:::{_FEATURES_BUCKET}",
                    f"arn:aws:s3:::{_FEATURES_BUCKET}/*",
                    f"arn:aws:s3:::{_ATHENA_RESULTS_BUCKET}",
                    f"arn:aws:s3:::{_ATHENA_RESULTS_BUCKET}/*",
                ],
            )
        )

        # Glue catalog: Athena resolves the Iceberg table via Glue.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "glue:GetDatabase",
                    "glue:GetDatabases",
                    "glue:GetTable",
                    "glue:GetTables",
                    "glue:GetTableVersion",
                    "glue:GetTableVersions",
                    "glue:GetPartition",
                    "glue:GetPartitions",
                    "glue:BatchGetPartition",
                ],
                resources=[
                    f"arn:aws:glue:{self.region}:{self.account}:catalog",
                    f"arn:aws:glue:{self.region}:{self.account}:database/{_GLUE_DATABASE}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{_GLUE_DATABASE}/{_GLUE_TABLE}",
                ],
            )
        )

        # Athena: run queries via awswrangler.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:GetQueryResultsStream",
                    "athena:StopQueryExecution",
                    "athena:GetWorkGroup",
                    "athena:GetDataCatalog",
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/{_ATHENA_WORKGROUP}",
                    f"arn:aws:athena:{self.region}:{self.account}:datacatalog/*",
                ],
            )
        )

        task_definition = ecs.FargateTaskDefinition(
            self,
            "InferenceFeaturesTaskDef",
            family="planequery-dev-inference-features",
            cpu=8192,                # 8 vCPU
            memory_limit_mib=16384,  # 16 GiB
            ephemeral_storage_gib=100,
            task_role=task_role,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.ARM64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )

        task_definition.add_container(
            "planequery-dev-inference-features",
            image=ecs.ContainerImage.from_docker_image_asset(image_asset),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="planequery-dev-inference-features",
                log_group=log_group,
            ),
            environment={
                "OUTPUT_DIR": "/app",
                "AWS_REGION": self.region,
                "ATHENA_WORKGROUP": _ATHENA_WORKGROUP,
            },
        )

        # ------------------------------------------------------------------
        # Step Functions: one Fargate task per day with bounded concurrency.
        # Map iterates items shaped like {"start": "YYYY-MM-DD",
        # "end": "YYYY-MM-DD"} (end is exclusive, so end == start+1).
        # ------------------------------------------------------------------
        sfn_log_group = logs.LogGroup(
            self,
            "InferenceFeaturesSfnLogs",
            log_group_name="/planequery/dev/inference-features-sfn",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        sfn_role = iam.Role(
            self,
            "InferenceFeaturesSfnRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Execution role for inference-features Step Functions state machine",
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

        run_task_state = sfn_tasks.EcsRunTask(
            self,
            "RunInferenceFeaturesTask",
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
                    command=sfn.JsonPath.list_at("$.command"),
                )
            ],
            propagated_tag_source=ecs.PropagatedTagSource.TASK_DEFINITION,
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            result_path=sfn.JsonPath.DISCARD,
        )

        map_state = sfn.Map(
            self,
            "InferenceFeaturesDateMap",
            comment="Process each date as a Fargate task with bounded concurrency.",
            items_path=sfn.JsonPath.string_at("$.dates"),
            max_concurrency_path=sfn.JsonPath.string_at("$.max_concurrency"),
            result_path=sfn.JsonPath.DISCARD,
        )
        map_state.item_processor(run_task_state)

        state_machine = sfn.StateMachine(
            self,
            "InferenceFeaturesStateMachine",
            state_machine_name="planequery-dev-inference-features",
            definition_body=sfn.DefinitionBody.from_chainable(map_state),
            role=sfn_role,
            state_machine_type=sfn.StateMachineType.STANDARD,
            logs=sfn.LogOptions(
                destination=sfn_log_group,
                level=sfn.LogLevel.ERROR,
                include_execution_data=False,
            ),
        )
        self.state_machine = state_machine

        # ------------------------------------------------------------------
        # Outputs consumed by scripts/run_inference_features_on_aws.py
        # ------------------------------------------------------------------
        CfnOutput(self, "ClusterName", value=cluster.cluster_name)
        CfnOutput(self, "TaskDefinitionArn", value=task_definition.task_definition_arn)
        CfnOutput(self, "ContainerName", value="planequery-dev-inference-features")
        CfnOutput(self, "SecurityGroupId", value=security_group.security_group_id)
        CfnOutput(
            self,
            "SubnetIds",
            value=",".join(s.subnet_id for s in vpc.public_subnets),
        )
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
        CfnOutput(self, "StateMachineArn", value=state_machine.state_machine_arn)
