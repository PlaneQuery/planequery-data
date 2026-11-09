from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
    aws_athena as athena,
    aws_glue as glue,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


_ENVIRONMENT_NAMES = ("dev", "prod")
_ATHENA_QUERY_RESULTS_PREFIX = "results/"
_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX = "planequery-api-athena-results/"
_PLANEQUERY_API_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY = 1 * 1024 * 1024 * 1024
_LEGACY_GLUE_DATABASE_LOGICAL_IDS = {
    "dev": "PlaneQueryDevAdsbGlueDatabase",
}
_LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS = {
    "dev": "GlueTableOptimizerRole65E74537",
}
_PLANEQUERY_PROD_WEBSITE_IAM_USER_ARN = (
    "arn:aws:iam::223281837468:user/planequery-prod-website"
)


class PlanequeryDatalakeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        self.buckets = {}
        self.athena_results_buckets = {}
        self.workgroups = {}

        for environment_name in _ENVIRONMENT_NAMES:
            self._create_datalake_environment(environment_name)

    def _create_datalake_environment(self, environment_name: str) -> None:
        label = environment_name.capitalize()
        resource_id = lambda suffix: f"PlaneQuery{label}{suffix}"
        resource_name = lambda suffix: f"planequery-{environment_name}-{suffix}"

        lake_bucket_name = resource_name("lake")
        glue_database_name = f"planequery_lake_{environment_name}"
        athena_results_bucket_name = resource_name("athena-results")
        athena_workgroup_name = f"planequery-workgroup-{environment_name}"

        bucket = s3.Bucket(
            self,
            resource_id("LakeBucket"),
            bucket_name=lake_bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )
        self.buckets[environment_name] = bucket

        database = glue.CfnDatabase(
            self,
            resource_id("GlueDatabase"),
            catalog_id=self.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(
                name=glue_database_name,
            ),
        )
        database.apply_removal_policy(RemovalPolicy.RETAIN)
        if environment_name in _LEGACY_GLUE_DATABASE_LOGICAL_IDS:
            database.override_logical_id(
                _LEGACY_GLUE_DATABASE_LOGICAL_IDS[environment_name]
            )

        athena_results_bucket = s3.Bucket(
            self,
            resource_id("AthenaResultsBucket"),
            bucket_name=athena_results_bucket_name,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )
        self.athena_results_buckets[environment_name] = athena_results_bucket

        workgroup = athena.CfnWorkGroup(
            self,
            resource_id("AthenaWorkGroup"),
            name=athena_workgroup_name,
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=(
                        f"s3://{athena_results_bucket.bucket_name}/"
                        f"{_ATHENA_QUERY_RESULTS_PREFIX}"
                    )
                ),
                enforce_work_group_configuration=True,
            ),
        )
        self.workgroups[environment_name] = workgroup

        # Iceberg table creation remains out-of-band. The optimizer role is
        # flights-only and cannot access the retired ADS-B table or prefix.
        optimizer_role = iam.Role(
            self,
            resource_id("GlueTableOptimizerRole"),
            role_name=resource_name("glue-table-optimizer"),
            assumed_by=iam.ServicePrincipal("glue.amazonaws.com"),
            inline_policies={
                "GlueTableOptimizerPolicy": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "glue:GetTable",
                                "glue:UpdateTable",
                                "glue:GetDatabase",
                            ],
                            resources=[
                                f"arn:aws:glue:{self.region}:{self.account}:catalog",
                                f"arn:aws:glue:{self.region}:{self.account}:database/{glue_database_name}",
                                f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/flights",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=["s3:GetBucketLocation", "s3:ListBucket"],
                            resources=[bucket.bucket_arn],
                            conditions={
                                "StringLike": {
                                    "s3:prefix": [
                                        "iceberg/flights",
                                        "iceberg/flights/*",
                                    ]
                                }
                            },
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                            ],
                            resources=[f"{bucket.bucket_arn}/iceberg/flights/*"],
                        ),
                    ]
                )
            },
        )
        if environment_name in _LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS:
            optimizer_role.node.default_child.override_logical_id(
                _LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS[environment_name]
            )

        CfnOutput(
            self,
            resource_id("GlueTableOptimizerRoleArn"),
            value=optimizer_role.role_arn,
        )

        # Only the production website retains external query access. Customer
        # IAM users, premium provisioning, and standalone API-assumer users are
        # deliberately absent from this stack.
        if environment_name != "prod":
            return

        api_athena_workgroup_name = resource_name("api-athena-workgroup")
        api_athena_role_name = resource_name("api-athena-readonly")

        planequery_api_athena_workgroup = athena.CfnWorkGroup(
            self,
            resource_id("ApiAthenaWorkGroup"),
            name=api_athena_workgroup_name,
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=(
                        f"s3://{athena_results_bucket.bucket_name}/"
                        f"{_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX}"
                    )
                ),
                enforce_work_group_configuration=True,
                bytes_scanned_cutoff_per_query=(
                    _PLANEQUERY_API_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY
                ),
            ),
        )

        planequery_api_athena_role = iam.Role(
            self,
            resource_id("ApiAthenaReadOnlyRole"),
            role_name=api_athena_role_name,
            assumed_by=iam.ArnPrincipal(_PLANEQUERY_PROD_WEBSITE_IAM_USER_ARN),
            description=(
                "Role for the PlaneQuery production website API to query the "
                "flights table through Athena."
            ),
        )

        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:StartQueryExecution",
                    "athena:GetQueryExecution",
                    "athena:BatchGetQueryExecution",
                    "athena:GetQueryResults",
                    "athena:GetQueryResultsStream",
                    "athena:GetQueryRuntimeStatistics",
                    "athena:StopQueryExecution",
                    "athena:GetWorkGroup",
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/{api_athena_workgroup_name}",
                ],
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:GetDataCatalog",
                    "athena:GetTableMetadata",
                    "athena:ListDatabases",
                    "athena:ListTableMetadata",
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:datacatalog/AwsDataCatalog",
                ],
            )
        )
        planequery_api_athena_role.add_to_policy(
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
                    f"arn:aws:glue:{self.region}:{self.account}:database/{glue_database_name}",
                    f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/flights",
                ],
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetBucketLocation", "s3:ListBucket"],
                resources=[bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": ["iceberg/flights", "iceberg/flights/*"]
                    }
                },
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"{bucket.bucket_arn}/iceberg/flights/*"],
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetBucketLocation"],
                resources=[athena_results_bucket.bucket_arn],
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[athena_results_bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            _PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX,
                            f"{_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX}*",
                        ]
                    }
                },
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[
                    f"{athena_results_bucket.bucket_arn}/"
                    f"{_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX}*"
                ],
            )
        )

        CfnOutput(
            self,
            resource_id("ApiAthenaReadOnlyRoleArn"),
            value=planequery_api_athena_role.role_arn,
        )
        CfnOutput(
            self,
            resource_id("ApiAthenaWorkGroupName"),
            value=planequery_api_athena_workgroup.name,
        )


# Run scripts/create_iceberg_table.py after deploying this stack.
