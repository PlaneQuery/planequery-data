"""AWS Batch stack for flight reconstruction and Iceberg ingest.

The stack provisions:

* An AMD64 Docker image built from ``Dockerfile.inference``.
* A managed AWS Batch EC2 compute environment with memory-optimized instances.
* A job queue and small/medium/large job definitions that run
  ``flights_from_model.py``.
* IAM permissions to read/write PlaneQuery S3 data and update the Glue
  Iceberg catalog.

Use ``scripts/run_flight_inference_batch_on_aws.py`` to submit and monitor jobs.
"""

from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_batch as batch,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct


_REPO_ROOT = Path(__file__).resolve().parents[2]

_DATA_BUCKETS = ("planequery-dev-lake", "planequery-prod-lake")
_PLANEQUERY_DATA_BUCKET = "planequery-dev-planequery-data"
_GLUE_DATABASE = "planequery_lake_prod"
_ADSB_TABLE = "adsb_messages"
_FLIGHTS_TABLE = "flights"
_MAX_PARALLEL_JOBS = 10
_JOB_QUEUE_NAME = "planequery-dev-flight-algorithm"
_COMPUTE_INSTANCE_TYPES = [
    "m7i.4xlarge",
    "m7i.8xlarge",
    "m7i.16xlarge",
]

_JOB_SIZE_CONFIGS = {
    "small": {
        "id_suffix": "Small",
        "name": "planequery-dev-flight-inference-small",
        "vcpus": 16,
        "memory_mib": 61440,
    },
    "medium": {
        "id_suffix": "Medium",
        "name": "planequery-dev-flight-inference-medium",
        "vcpus": 32,
        "memory_mib": 122880,
    },
    "large": {
        "id_suffix": "Large",
        "name": "planequery-dev-flight-inference-large",
        "vcpus": 64,
        "memory_mib": 245760,
    },
}

# Keep the compute cap at ten 16-vCPU job equivalents. Larger jobs can still run,
# but fewer can run concurrently under the same cap.
_COMPUTE_ENV_MAX_VCPUS = 32 * _MAX_PARALLEL_JOBS


class PlanequeryFlightInferenceBatchStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ------------------------------------------------------------------
        # Networking: use default VPC public subnets so Batch EC2 instances
        # can reach ECR, S3, Athena, Glue, and Maven without a NAT gateway.
        # ------------------------------------------------------------------
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        security_group = ec2.SecurityGroup(
            self,
            "FlightInferenceBatchSg",
            vpc=vpc,
            description="Flight inference AWS Batch instances (egress only)",
            allow_all_outbound=True,
        )

        # ------------------------------------------------------------------
        # Build and push the AMD64 algorithm image to ECR.
        # ------------------------------------------------------------------
        image_asset = ecr_assets.DockerImageAsset(
            self,
            "FlightInferenceImage",
            directory=str(_REPO_ROOT),
            file="Dockerfile.inference",
            platform=ecr_assets.Platform.LINUX_AMD64,
            exclude=[
                "data/raw/**",
                "data/models/**",
                "notebooks/**",
                "tests/**",
                "cdk/cdk.out/**",
                ".venv/**",
                ".git/**",
                "**/__pycache__/**",
            ],
        )

        # ------------------------------------------------------------------
        # Logs
        # ------------------------------------------------------------------
        log_group = logs.LogGroup(
            self,
            "FlightInferenceBatchLogs",
            log_group_name="/planequery/dev/flight-inference-batch/jobs",
            retention=logs.RetentionDays.ONE_MONTH,
        )

        # ------------------------------------------------------------------
        # Roles
        # ------------------------------------------------------------------
        batch_service_role = iam.Role(
            self,
            "FlightInferenceBatchServiceRole",
            assumed_by=iam.ServicePrincipal("batch.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSBatchServiceRole"
                )
            ],
        )

        instance_role = iam.Role(
            self,
            "FlightInferenceBatchInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonEC2ContainerRegistryReadOnly"
                ),
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                ),
            ],
        )
        instance_profile = iam.CfnInstanceProfile(
            self,
            "FlightInferenceBatchInstanceProfile",
            roles=[instance_role.role_name],
        )

        execution_role = iam.Role(
            self,
            "FlightInferenceBatchExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )

        job_role = iam.Role(
            self,
            "FlightInferenceBatchJobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="Runtime role for flight reconstruction and Iceberg ingest jobs",
        )

        image_asset.repository.grant_pull(instance_role)
        image_asset.repository.grant_pull(execution_role)

        # S3: read/write the Iceberg lake buckets, read/write the planequery-data
        # bucket for ADS-B, model, and published flight artifacts.
        # Prod flight inference writes planequery_lake_prod.flights in
        # planequery-prod-lake, while shared ADS-B inputs still live in
        # planequery-dev-lake.
        job_role.add_to_policy(
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
                    *[
                        resource
                        for bucket in _DATA_BUCKETS
                        for resource in (
                            f"arn:aws:s3:::{bucket}",
                            f"arn:aws:s3:::{bucket}/*",
                        )
                    ],
                    f"arn:aws:s3:::{_PLANEQUERY_DATA_BUCKET}",
                    f"arn:aws:s3:::{_PLANEQUERY_DATA_BUCKET}/*",
                ],
            )
        )

        # Glue catalog: read ADS-B/flight tables and update Iceberg metadata
        # when Spark appends inferred flights.
        job_role.add_to_policy(
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
                    f"arn:aws:glue:{self.region}:{self.account}:table/{_GLUE_DATABASE}/{_ADSB_TABLE}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{_GLUE_DATABASE}/{_FLIGHTS_TABLE}",
                ],
            )
        )

        # ------------------------------------------------------------------
        # AWS Batch EC2 compute environment using EC2 Spot instances and a
        # set of allowed memory-optimized instance types. Using Spot reduces cost and
        # allows Batch to pick any of the listed instance sizes.
        # Each job definition sets memory below the matching host total to leave room
        # for ECS and OS overhead on the instance.
        # ------------------------------------------------------------------
        # Create an EC2 Launch Template to set a larger root EBS volume (100 GiB)
        launch_template = ec2.CfnLaunchTemplate(
            self,
            "FlightInferenceLaunchTemplate",
            launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                block_device_mappings=[
                    ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                        device_name="/dev/xvda",
                        ebs=ec2.CfnLaunchTemplate.EbsProperty(
                            delete_on_termination=True,
                            volume_size=100,
                            volume_type="gp3",
                        ),
                    )
                ]
            ),
        )

        compute_environment = batch.CfnComputeEnvironment(
            self,
            "FlightInferenceComputeEnvironment",
            compute_environment_name=(
                "planequery-dev-flight-algorithm-spot-v1"
            ),
            type="MANAGED",
            state="ENABLED",
            service_role=batch_service_role.role_arn,
            compute_resources=batch.CfnComputeEnvironment.ComputeResourcesProperty(
                type="SPOT",
                allocation_strategy="SPOT_CAPACITY_OPTIMIZED",
                minv_cpus=0,
                maxv_cpus=_COMPUTE_ENV_MAX_VCPUS,
                instance_types=_COMPUTE_INSTANCE_TYPES,
                instance_role=instance_profile.attr_arn,
                subnets=[s.subnet_id for s in vpc.public_subnets],
                security_group_ids=[security_group.security_group_id],
                ec2_configuration=[
                    batch.CfnComputeEnvironment.Ec2ConfigurationObjectProperty(
                        image_type="ECS_AL2023",
                    )
                ],
                launch_template=batch.CfnComputeEnvironment.LaunchTemplateSpecificationProperty(
                    launch_template_id=launch_template.ref,
                    version="$Latest",
                ),
            ),
        )

        job_queue = batch.CfnJobQueue(
            self,
            "FlightInferenceJobQueue",
            job_queue_name=_JOB_QUEUE_NAME,
            priority=1,
            state="ENABLED",
            compute_environment_order=[
                batch.CfnJobQueue.ComputeEnvironmentOrderProperty(
                    order=1,
                    compute_environment=compute_environment.ref,
                )
            ],
        )

        def create_job_definition(size: str, config: dict[str, int | str]) -> batch.CfnJobDefinition:
            id_suffix = config["id_suffix"]
            return batch.CfnJobDefinition(
                self,
                f"FlightInference{id_suffix}JobDefinition",
                job_definition_name=str(config["name"]),
                type="container",
                platform_capabilities=["EC2"],
                retry_strategy=batch.CfnJobDefinition.RetryStrategyProperty(
                    attempts=8,
                    evaluate_on_exit=[
                        batch.CfnJobDefinition.EvaluateOnExitProperty(
                            on_status_reason="Host EC2*",
                            action="RETRY",
                        ),
                        batch.CfnJobDefinition.EvaluateOnExitProperty(
                            on_exit_code="*",
                            action="EXIT",
                        ),
                    ],
                ),
                timeout=batch.CfnJobDefinition.TimeoutProperty(attempt_duration_seconds=86400),
                container_properties=batch.CfnJobDefinition.ContainerPropertiesProperty(
                    image=image_asset.image_uri,
                    command=[
                        "python",
                        "/app/src/data_engineering/flights/flights_from_model.py",
                    ],
                    job_role_arn=job_role.role_arn,
                    execution_role_arn=execution_role.role_arn,
                    resource_requirements=[
                        batch.CfnJobDefinition.ResourceRequirementProperty(
                            type="VCPU",
                            value=str(config["vcpus"]),
                        ),
                        batch.CfnJobDefinition.ResourceRequirementProperty(
                            type="MEMORY",
                            value=str(config["memory_mib"]),
                        ),
                    ],
                    environment=[
                        batch.CfnJobDefinition.EnvironmentProperty(
                            name="OUTPUT_DIR",
                            value="/app",
                        ),
                        batch.CfnJobDefinition.EnvironmentProperty(
                            name="AWS_REGION",
                            value=self.region,
                        ),
                        batch.CfnJobDefinition.EnvironmentProperty(
                            name="PLANEQUERY_BATCH_JOB_SIZE",
                            value=size,
                        ),
                        batch.CfnJobDefinition.EnvironmentProperty(
                            name="PLANEQUERY_READ_FROM_S3",
                            value="1",
                        ),
                    ],
                    log_configuration=batch.CfnJobDefinition.LogConfigurationProperty(
                        log_driver="awslogs",
                        options={
                            "awslogs-group": log_group.log_group_name,
                            "awslogs-region": self.region,
                            "awslogs-stream-prefix": f"planequery-dev-flight-inference-{size}",
                        },
                    ),
                ),
            )

        job_definitions = {
            size: create_job_definition(size, config)
            for size, config in _JOB_SIZE_CONFIGS.items()
        }
        self.job_queue_arn = (
            f"arn:{self.partition}:batch:{self.region}:{self.account}:"
            f"job-queue/{_JOB_QUEUE_NAME}"
        )
        self.job_queue_name = _JOB_QUEUE_NAME
        self.job_definition_arn = job_definitions["small"].ref
        self.job_definition_name = str(_JOB_SIZE_CONFIGS["small"]["name"])
        self.max_parallel_jobs = _MAX_PARALLEL_JOBS

        # ------------------------------------------------------------------
        # Outputs consumed by scripts/run_flight_inference_batch_on_aws.py
        # ------------------------------------------------------------------
        CfnOutput(self, "ComputeEnvironmentName", value=compute_environment.ref)
        CfnOutput(self, "JobQueueName", value=job_queue.ref)
        CfnOutput(self, "JobQueueArn", value=job_queue.attr_job_queue_arn)
        CfnOutput(self, "JobDefinitionArn", value=job_definitions["small"].ref)
        CfnOutput(self, "JobDefinitionName", value=str(_JOB_SIZE_CONFIGS["small"]["name"]))
        for size, job_definition in job_definitions.items():
            id_suffix = _JOB_SIZE_CONFIGS[size]["id_suffix"]
            CfnOutput(self, f"JobDefinition{id_suffix}Arn", value=job_definition.ref)
            CfnOutput(
                self,
                f"JobDefinition{id_suffix}Name",
                value=str(_JOB_SIZE_CONFIGS[size]["name"]),
            )
        CfnOutput(self, "LogGroupName", value=log_group.log_group_name)
        CfnOutput(self, "MaxParallelJobs", value=str(_MAX_PARALLEL_JOBS))
