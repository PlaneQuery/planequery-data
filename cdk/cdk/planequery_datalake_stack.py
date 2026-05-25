import os

from aws_cdk import (
    CfnOutput,
    Stack,
    RemovalPolicy,
    SecretValue,
    aws_s3 as s3,
    aws_glue as glue,
    aws_athena as athena,
    aws_iam as iam,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


_ENVIRONMENT_NAMES = ("dev", "prod")
_ATHENA_QUERY_RESULTS_PREFIX = "results/"
_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX = "planequery-api-athena-results/"
_PLANEQUERY_API_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY = 1 * 1024 * 1024 * 1024
_PREMIUM_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY = 800 * 1024 * 1024 * 1024
_LEGACY_GLUE_DATABASE_LOGICAL_IDS = {
    "dev": "PlaneQueryDevAdsbGlueDatabase",
}
_LEGACY_MANAGED_POLICY_LOGICAL_IDS = {
    "prod": "PlaneQueryPremiumCustomerReadOnlyPolicyBDB2E5BA",
}
_LEGACY_ROLE_LOGICAL_IDS = {
    "prod": "PlaneQueryPremiumProvisionerRole54B93886",
}
_LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS = {
    "dev": "GlueTableOptimizerRole65E74537",
}
_LEGACY_ROLE_DEFAULT_POLICY_LOGICAL_IDS = {
    "prod": "PlaneQueryPremiumProvisionerRoleDefaultPolicyA0A1B93D",
}
_PLANEQUERY_PROD_WEBSITE_IAM_USER_ARN = (
    "arn:aws:iam::223281837468:user/planequery-prod-website"
)
_CROSS_ENVIRONMENT_LAKE_READ_BUCKETS = {
    # The prod Glue catalog can point at Iceberg tables migrated from dev while
    # preserving their existing metadata/data locations.
    "prod": ("planequery-dev-lake",),
}


def _premium_customer_prefix(environment_name: str) -> str:
    if environment_name == "prod":
        return "planequery-premium-customer-"
    return f"planequery-{environment_name}-premium-customer-"


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
        api_athena_workgroup_name = resource_name("api-athena-workgroup")
        api_athena_role_name = resource_name("api-athena-readonly")
        vercel_api_athena_user_name = resource_name("vercel-api-athena-assumer")
        vercel_api_athena_secret_name = (
            f"planequery/{environment_name}/vercel-api-athena-assumer"
        )
        premium_customer_user_prefix = _premium_customer_prefix(environment_name)
        premium_customer_workgroup_prefix = _premium_customer_prefix(environment_name)
        premium_customer_results_prefix = resource_name("premium-results/customer/")
        premium_customer_readonly_policy_name = resource_name("premium-athena-readonly")
        premium_provisioner_role_name = resource_name("premium-provisioner")
        optimizer_lake_bucket_names = (
            lake_bucket_name,
            *_CROSS_ENVIRONMENT_LAKE_READ_BUCKETS.get(environment_name, ()),
        )

        # S3 bucket for Iceberg data. Data buckets are retained on stack delete.
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

        # Glue database (Iceberg catalog via Athena). Databases are retained on
        # stack delete to protect table metadata.
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

        # Athena results bucket. Query result buckets are also retained on
        # stack delete because they are S3 buckets owned by the environment.
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

        # Athena workgroup
        workgroup = athena.CfnWorkGroup(
            self,
            resource_id("AthenaWorkGroup"),
            name=athena_workgroup_name,
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{athena_results_bucket.bucket_name}/{_ATHENA_QUERY_RESULTS_PREFIX}"
                ),
                enforce_work_group_configuration=True,
            ),
        )
        self.workgroups[environment_name] = workgroup

        # Dedicated workgroup for the public API. It keeps query outputs in a
        # separate prefix so the API role does not need access to every Athena
        # result object produced by other workloads.
        planequery_api_athena_workgroup = athena.CfnWorkGroup(
            self,
            resource_id("ApiAthenaWorkGroup"),
            name=api_athena_workgroup_name,
            recursive_delete_option=True,
            work_group_configuration=athena.CfnWorkGroup.WorkGroupConfigurationProperty(
                result_configuration=athena.CfnWorkGroup.ResultConfigurationProperty(
                    output_location=f"s3://{athena_results_bucket.bucket_name}/{_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX}"
                ),
                enforce_work_group_configuration=True,
                bytes_scanned_cutoff_per_query=_PLANEQUERY_API_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY,
            ),
        )

        # Note: Iceberg table creation is handled out-of-band (e.g. via
        # scripts/create_iceberg_table.py) instead of from this stack.

        # IAM role for Glue managed table optimizers
        # Note: table optimizers are configured out-of-band via
        # scripts/create_iceberg_table.py after the Iceberg table is created.
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
                                f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/adsb_messages",
                                f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/flights",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                                "s3:ListBucket",
                            ],
                            resources=[
                                *[
                                    f"arn:aws:s3:::{bucket_name}"
                                    for bucket_name in optimizer_lake_bucket_names
                                ],
                                *[
                                    f"arn:aws:s3:::{bucket_name}/*"
                                    for bucket_name in optimizer_lake_bucket_names
                                ],
                            ],
                        ),
                    ]
                )
            },
        )
        if environment_name in _LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS:
            optimizer_role.node.default_child.override_logical_id(
                _LEGACY_GLUE_TABLE_OPTIMIZER_ROLE_LOGICAL_IDS[environment_name]
            )

        # IAM cannot enforce a total Athena row limit or SELECT-only SQL.
        # Enforce that in the PlaneQuery API by validating queries, appending
        # LIMIT 1000 where appropriate, and not paginating past the first
        # GetQueryResults page.
        # Minimal long-lived IAM user for Vercel. Its only permission is to
        # assume the read-only Athena role below; it has no direct Athena,
        # Glue, or S3 access. The generated access key is stored in Secrets
        # Manager for one-time copy into Vercel environment variables.
        vercel_api_athena_assumer_user = iam.User(
            self,
            resource_id("VercelApiAthenaAssumerUser"),
            user_name=vercel_api_athena_user_name,
        )
        planequery_api_athena_trusted_principals = [
            iam.ArnPrincipal(vercel_api_athena_assumer_user.user_arn)
        ]
        if environment_name == "prod":
            planequery_api_athena_trusted_principals.append(
                iam.ArnPrincipal(_PLANEQUERY_PROD_WEBSITE_IAM_USER_ARN)
            )
        environment_specific_trusted_principal_env = (
            f"PLANEQUERY_{environment_name.upper()}_API_ATHENA_TRUSTED_PRINCIPAL_ARN"
        )
        planequery_api_athena_extra_trusted_principal_arn = os.environ.get(
            environment_specific_trusted_principal_env
        )
        if planequery_api_athena_extra_trusted_principal_arn:
            planequery_api_athena_trusted_principals.append(
                iam.ArnPrincipal(planequery_api_athena_extra_trusted_principal_arn)
            )
        planequery_api_athena_assumed_by = iam.CompositePrincipal(
            *planequery_api_athena_trusted_principals
        )

        planequery_api_athena_role = iam.Role(
            self,
            resource_id("ApiAthenaReadOnlyRole"),
            role_name=api_athena_role_name,
            assumed_by=planequery_api_athena_assumed_by,
            description=(
                f"Role for the PlaneQuery {environment_name} API to run read-only "
                f"Athena queries against {glue_database_name}"
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
                    f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/*",
                ],
            )
        )

        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetBucketLocation",
                    "s3:ListBucket",
                ],
                resources=[bucket.bucket_arn],
            )
        )
        planequery_api_athena_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[f"{bucket.bucket_arn}/*"],
            )
        )
        for shared_lake_bucket_name in _CROSS_ENVIRONMENT_LAKE_READ_BUCKETS.get(
            environment_name, ()
        ):
            shared_lake_bucket_arn = f"arn:aws:s3:::{shared_lake_bucket_name}"
            planequery_api_athena_role.add_to_policy(
                iam.PolicyStatement(
                    actions=[
                        "s3:GetBucketLocation",
                        "s3:ListBucket",
                    ],
                    resources=[shared_lake_bucket_arn],
                    conditions={
                        "StringLike": {
                            "s3:prefix": [
                                "iceberg/",
                                "iceberg/*",
                            ]
                        }
                    },
                )
            )
            planequery_api_athena_role.add_to_policy(
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[f"{shared_lake_bucket_arn}/iceberg/*"],
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
                    f"{athena_results_bucket.bucket_arn}/{_PLANEQUERY_API_ATHENA_QUERY_RESULTS_PREFIX}*"
                ],
            )
        )

        # Managed policy attached to each generated premium customer IAM user.
        # Athena query access is restricted to the workgroup named after the
        # IAM username, and S3 result access is restricted by IAM username.
        premium_customer_readonly_policy = iam.ManagedPolicy(
            self,
            resource_id("PremiumCustomerReadOnlyPolicy"),
            managed_policy_name=premium_customer_readonly_policy_name,
            statements=[
                iam.PolicyStatement(
                    actions=[
                        "athena:StartQueryExecution",
                        "athena:GetQueryExecution",
                        "athena:GetQueryResults",
                        "athena:GetQueryResultsStream",
                        "athena:GetQueryRuntimeStatistics",
                        "athena:StopQueryExecution",
                        "athena:GetWorkGroup",
                    ],
                    resources=[
                        f"arn:aws:athena:{self.region}:{self.account}:workgroup/${{aws:username}}",
                    ],
                ),
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
                ),
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
                        f"arn:aws:glue:{self.region}:{self.account}:table/{glue_database_name}/*",
                    ],
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:GetBucketLocation",
                        "s3:ListBucket",
                    ],
                    resources=[bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    actions=["s3:GetObject"],
                    resources=[f"{bucket.bucket_arn}/*"],
                ),
                *[
                    iam.PolicyStatement(
                        actions=[
                            "s3:GetBucketLocation",
                            "s3:ListBucket",
                        ],
                        resources=[f"arn:aws:s3:::{shared_lake_bucket_name}"],
                        conditions={
                            "StringLike": {
                                "s3:prefix": [
                                    "iceberg/",
                                    "iceberg/*",
                                ]
                            }
                        },
                    )
                    for shared_lake_bucket_name in _CROSS_ENVIRONMENT_LAKE_READ_BUCKETS.get(
                        environment_name, ()
                    )
                ],
                *[
                    iam.PolicyStatement(
                        actions=["s3:GetObject"],
                        resources=[
                            f"arn:aws:s3:::{shared_lake_bucket_name}/iceberg/*"
                        ],
                    )
                    for shared_lake_bucket_name in _CROSS_ENVIRONMENT_LAKE_READ_BUCKETS.get(
                        environment_name, ()
                    )
                ],
                iam.PolicyStatement(
                    actions=["s3:GetBucketLocation"],
                    resources=[athena_results_bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    actions=["s3:ListBucket"],
                    resources=[athena_results_bucket.bucket_arn],
                    conditions={
                        "StringLike": {
                            "s3:prefix": [
                                f"{premium_customer_results_prefix}${{aws:username}}/",
                                f"{premium_customer_results_prefix}${{aws:username}}/*",
                            ]
                        }
                    },
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:AbortMultipartUpload",
                        "s3:ListMultipartUploadParts",
                    ],
                    resources=[
                        f"{athena_results_bucket.bucket_arn}/{premium_customer_results_prefix}${{aws:username}}/*"
                    ],
                ),
            ],
        )
        if environment_name in _LEGACY_MANAGED_POLICY_LOGICAL_IDS:
            premium_customer_readonly_policy.node.default_child.override_logical_id(
                _LEGACY_MANAGED_POLICY_LOGICAL_IDS[environment_name]
            )

        premium_provisioner_trusted_principals = [
            iam.ArnPrincipal(vercel_api_athena_assumer_user.user_arn)
        ]
        if environment_name == "prod":
            premium_provisioner_trusted_principals.append(
                iam.ArnPrincipal(_PLANEQUERY_PROD_WEBSITE_IAM_USER_ARN)
            )

        premium_provisioner_role = iam.Role(
            self,
            resource_id("PremiumProvisionerRole"),
            role_name=premium_provisioner_role_name,
            assumed_by=iam.CompositePrincipal(*premium_provisioner_trusted_principals),
            description=(
                f"Role used by PlaneQuery {environment_name} API to create "
                "isolated premium Athena customers."
            ),
        )
        if environment_name in _LEGACY_ROLE_LOGICAL_IDS:
            premium_provisioner_role.node.default_child.override_logical_id(
                _LEGACY_ROLE_LOGICAL_IDS[environment_name]
            )

        premium_provisioner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "iam:CreateUser",
                    "iam:GetUser",
                    "iam:TagUser",
                    "iam:DeleteUser",
                    "iam:CreateAccessKey",
                    "iam:ListAccessKeys",
                    "iam:UpdateAccessKey",
                    "iam:DeleteAccessKey",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:user/{premium_customer_user_prefix}*"
                ],
            )
        )

        premium_provisioner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "iam:AttachUserPolicy",
                    "iam:DetachUserPolicy",
                    "iam:ListAttachedUserPolicies",
                ],
                resources=[
                    f"arn:aws:iam::{self.account}:user/{premium_customer_user_prefix}*"
                ],
                conditions={
                    "ArnEquals": {
                        "iam:PolicyARN": premium_customer_readonly_policy.managed_policy_arn
                    }
                },
            )
        )

        premium_provisioner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "athena:CreateWorkGroup",
                    "athena:GetWorkGroup",
                    "athena:UpdateWorkGroup",
                    "athena:DeleteWorkGroup",
                    "athena:TagResource",
                    "athena:ListTagsForResource",
                ],
                resources=[
                    f"arn:aws:athena:{self.region}:{self.account}:workgroup/{premium_customer_workgroup_prefix}*"
                ],
            )
        )

        premium_provisioner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:PutObject",
                    "s3:GetObject",
                    "s3:DeleteObject",
                ],
                resources=[
                    f"{athena_results_bucket.bucket_arn}/{premium_customer_results_prefix}*"
                ],
            )
        )

        premium_provisioner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetBucketLocation",
                    "s3:ListBucket",
                ],
                resources=[athena_results_bucket.bucket_arn],
                conditions={
                    "StringLike": {
                        "s3:prefix": [
                            premium_customer_results_prefix,
                            f"{premium_customer_results_prefix}*",
                        ]
                    }
                },
            )
        )

        vercel_api_athena_assumer_user.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[
                    planequery_api_athena_role.role_arn,
                    premium_provisioner_role.role_arn,
                ],
            )
        )
        premium_provisioner_role_default_policy = premium_provisioner_role.node.try_find_child(
            "DefaultPolicy"
        )
        if (
            premium_provisioner_role_default_policy is not None
            and environment_name in _LEGACY_ROLE_DEFAULT_POLICY_LOGICAL_IDS
        ):
            premium_provisioner_role_default_policy.node.default_child.override_logical_id(
                _LEGACY_ROLE_DEFAULT_POLICY_LOGICAL_IDS[environment_name]
            )

        vercel_api_athena_access_key = iam.CfnAccessKey(
            self,
            resource_id("VercelApiAthenaAssumerAccessKey"),
            user_name=vercel_api_athena_assumer_user.user_name,
        )
        vercel_api_athena_credentials_secret = secretsmanager.Secret(
            self,
            resource_id("VercelApiAthenaCredentialsSecret"),
            secret_name=vercel_api_athena_secret_name,
            secret_object_value={
                "AWS_ACCESS_KEY_ID": SecretValue.unsafe_plain_text(
                    vercel_api_athena_access_key.ref
                ),
                "AWS_SECRET_ACCESS_KEY": SecretValue.unsafe_plain_text(
                    vercel_api_athena_access_key.attr_secret_access_key
                ),
                "AWS_REGION": SecretValue.unsafe_plain_text(self.region),
                "PLANEQUERY_API_ATHENA_ROLE_ARN": SecretValue.unsafe_plain_text(
                    planequery_api_athena_role.role_arn
                ),
                "ATHENA_WORKGROUP": SecretValue.unsafe_plain_text(
                    api_athena_workgroup_name
                ),
                "ATHENA_DATABASE": SecretValue.unsafe_plain_text(glue_database_name),
                "PLANEQUERY_PREMIUM_PROVISIONER_ROLE_ARN": SecretValue.unsafe_plain_text(
                    premium_provisioner_role.role_arn
                ),
                "PLANEQUERY_PREMIUM_CUSTOMER_READONLY_POLICY_ARN": SecretValue.unsafe_plain_text(
                    premium_customer_readonly_policy.managed_policy_arn
                ),
                "PREMIUM_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY": SecretValue.unsafe_plain_text(
                    str(_PREMIUM_ATHENA_BYTES_SCANNED_CUTOFF_PER_QUERY)
                ),
                "ATHENA_RESULTS_BUCKET": SecretValue.unsafe_plain_text(
                    athena_results_bucket.bucket_name
                ),
            },
        )

        CfnOutput(
            self,
            resource_id("GlueTableOptimizerRoleArn"),
            value=optimizer_role.role_arn,
        )
        CfnOutput(
            self,
            resource_id("ApiAthenaReadOnlyRoleArn"),
            value=planequery_api_athena_role.role_arn,
        )
        CfnOutput(
            self,
            resource_id("VercelApiAthenaAssumerUserArn"),
            value=vercel_api_athena_assumer_user.user_arn,
        )
        CfnOutput(
            self,
            resource_id("VercelApiAthenaCredentialsSecretArn"),
            value=vercel_api_athena_credentials_secret.secret_arn,
        )
        CfnOutput(
            self,
            resource_id("ApiAthenaWorkGroupName"),
            value=planequery_api_athena_workgroup.name,
        )
        CfnOutput(
            self,
            resource_id("PremiumProvisionerRoleArn"),
            value=premium_provisioner_role.role_arn,
        )
        CfnOutput(
            self,
            resource_id("PremiumCustomerReadOnlyPolicyArn"),
            value=premium_customer_readonly_policy.managed_policy_arn,
        )


# Run python scripts/create_iceberg_table.py after deploying this.
