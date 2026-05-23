from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    SecretValue,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
)
from constructs import Construct


_BUCKET_NAME = "planequery-dev-adsb-reapi"
_S3_PREFIX = "feeder-1/data/raw/reapi"
_CREDENTIALS_SECRET_NAME = "planequery/dev/adsb-realtime/credentials"
_USER_NAME = "planequery-dev-adsb-realtime-uploader"
_ROLE_NAME = "planequery-dev-adsb-realtime-uploader"
_POLICY_NAME = "planequery-dev-adsb-realtime-s3-access"


class PlanequeryAdsbRealtimeStack(Stack):
    """Resources for the machine running src/adsb/main.py in realtime."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        bucket = s3.Bucket(
            self,
            "AdsbReapiBucket",
            bucket_name=_BUCKET_NAME,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
            auto_delete_objects=False,
        )

        object_prefix = f"{_S3_PREFIX}/*"
        s3_access_policy = iam.ManagedPolicy(
            self,
            "AdsbRealtimeS3AccessPolicy",
            managed_policy_name=_POLICY_NAME,
            statements=[
                iam.PolicyStatement(
                    actions=["s3:GetBucketLocation"],
                    resources=[bucket.bucket_arn],
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:ListBucket",
                        "s3:ListBucketMultipartUploads",
                    ],
                    resources=[bucket.bucket_arn],
                    conditions={
                        "StringLike": {
                            "s3:prefix": [
                                _S3_PREFIX,
                                object_prefix,
                            ]
                        }
                    },
                ),
                iam.PolicyStatement(
                    actions=[
                        "s3:AbortMultipartUpload",
                        "s3:GetObject",
                        "s3:ListMultipartUploadParts",
                        "s3:PutObject",
                    ],
                    resources=[f"{bucket.bucket_arn}/{object_prefix}"],
                ),
            ],
        )

        uploader_user = iam.User(
            self,
            "AdsbRealtimeUploaderUser",
            user_name=_USER_NAME,
        )
        uploader_user.add_managed_policy(s3_access_policy)

        uploader_role = iam.Role(
            self,
            "AdsbRealtimeUploaderRole",
            role_name=_ROLE_NAME,
            assumed_by=iam.ArnPrincipal(
                f"arn:{self.partition}:iam::{self.account}:user/{_USER_NAME}"
            ),
            managed_policies=[s3_access_policy],
        )
        uploader_user.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[uploader_role.role_arn],
            )
        )

        access_key = iam.CfnAccessKey(
            self,
            "AdsbRealtimeUploaderAccessKey",
            user_name=uploader_user.user_name,
        )

        credentials_secret = secretsmanager.Secret(
            self,
            "AdsbRealtimeUploaderCredentialsSecret",
            secret_name=_CREDENTIALS_SECRET_NAME,
            secret_object_value={
                "AWS_ACCESS_KEY_ID": SecretValue.unsafe_plain_text(access_key.ref),
                "AWS_SECRET_ACCESS_KEY": SecretValue.unsafe_plain_text(
                    access_key.attr_secret_access_key
                ),
                "AWS_REGION": SecretValue.unsafe_plain_text(self.region),
                "ADSB_REAPI_S3_BUCKET": SecretValue.unsafe_plain_text(bucket.bucket_name),
                "ADSB_REAPI_S3_PREFIX": SecretValue.unsafe_plain_text(_S3_PREFIX),
                "ADSB_REAPI_UPLOADER_ROLE_ARN": SecretValue.unsafe_plain_text(
                    uploader_role.role_arn
                ),
            },
        )

        CfnOutput(self, "BucketName", value=bucket.bucket_name)
        CfnOutput(self, "S3Prefix", value=_S3_PREFIX)
        CfnOutput(self, "UploaderUserName", value=uploader_user.user_name)
        CfnOutput(self, "UploaderRoleArn", value=uploader_role.role_arn)
        CfnOutput(self, "CredentialsSecretName", value=credentials_secret.secret_name)
