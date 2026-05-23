"""Delete AWS resources for one PlaneQuery premium Athena customer.

This removes resources associated with a single premium IAM username, such as::

    planequery-prod-premium-customer-stripe-84a1049bec65fc0c

Deleted resources:
  - Athena workgroup named exactly the IAM username
  - S3 Athena result objects under the customer's result prefix
  - IAM access keys, attached policies, inline policies, group memberships, and user

The shared premium readonly managed policy is not deleted because it is reused by
all premium customers.

Usage::

    python scripts/delete_premium_customer.py \
        planequery-prod-premium-customer-stripe-84a1049bec65fc0c \
        --yes

Omit ``--yes`` to do a dry run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

import boto3
from botocore.exceptions import ClientError


DEFAULT_REGION = "us-east-2"
DEFAULT_SECRET_NAME = "planequery/dev/vercel-api-athena-assumer"
DEFAULT_ATHENA_RESULTS_BUCKET = "planequery-dev-athena-results"
PREMIUM_CUSTOMER_USER_PREFIX = "planequery-prod-premium-customer-"
PREMIUM_CUSTOMER_RESULTS_PREFIX = "planequery-prod-premium-results/customer/"


class DeletionPlan:
    def __init__(self, dry_run: bool) -> None:
        self.dry_run = dry_run

    def do(self, message: str, fn, *args, **kwargs) -> None:
        prefix = "DRY RUN" if self.dry_run else "DELETE"
        print(f"[{prefix}] {message}")
        if not self.dry_run:
            fn(*args, **kwargs)


def _client_error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def _ignore_not_found(exc: ClientError, not_found_codes: Iterable[str]) -> bool:
    return _client_error_code(exc) in set(not_found_codes)


def _load_secret_config(session: boto3.Session, secret_name: str) -> dict[str, str]:
    secretsmanager = session.client("secretsmanager")
    try:
        resp = secretsmanager.get_secret_value(SecretId=secret_name)
    except ClientError as exc:
        if _ignore_not_found(exc, {"ResourceNotFoundException"}):
            return {}
        raise

    secret_string = resp.get("SecretString")
    if not secret_string:
        return {}
    return json.loads(secret_string)


def _delete_s3_prefix(plan: DeletionPlan, session: boto3.Session, bucket: str, prefix: str) -> None:
    s3 = session.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    total = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if not objects:
            continue

        total += len(objects)
        for start in range(0, len(objects), 1000):
            batch = objects[start : start + 1000]
            plan.do(
                f"s3://{bucket}/{prefix}: delete {len(batch)} object(s)",
                s3.delete_objects,
                Bucket=bucket,
                Delete={"Objects": batch, "Quiet": True},
            )

    if total == 0:
        print(f"[SKIP] No S3 objects found under s3://{bucket}/{prefix}")


def _delete_athena_workgroup(plan: DeletionPlan, session: boto3.Session, workgroup: str) -> None:
    athena = session.client("athena")
    try:
        athena.get_work_group(WorkGroup=workgroup)
    except ClientError as exc:
        if _ignore_not_found(exc, {"InvalidRequestException", "ResourceNotFoundException"}):
            print(f"[SKIP] Athena workgroup does not exist: {workgroup}")
            return
        raise

    plan.do(
        f"Athena workgroup: {workgroup}",
        athena.delete_work_group,
        WorkGroup=workgroup,
        RecursiveDeleteOption=True,
    )


def _delete_iam_user(plan: DeletionPlan, session: boto3.Session, username: str) -> None:
    iam = session.client("iam")
    try:
        iam.get_user(UserName=username)
    except ClientError as exc:
        if _ignore_not_found(exc, {"NoSuchEntity"}):
            print(f"[SKIP] IAM user does not exist: {username}")
            return
        raise

    for page in iam.get_paginator("list_access_keys").paginate(UserName=username):
        for key in page.get("AccessKeyMetadata", []):
            plan.do(
                f"IAM access key: {key['AccessKeyId']}",
                iam.delete_access_key,
                UserName=username,
                AccessKeyId=key["AccessKeyId"],
            )

    for page in iam.get_paginator("list_attached_user_policies").paginate(UserName=username):
        for policy in page.get("AttachedPolicies", []):
            plan.do(
                f"Detach IAM policy: {policy['PolicyArn']}",
                iam.detach_user_policy,
                UserName=username,
                PolicyArn=policy["PolicyArn"],
            )

    for page in iam.get_paginator("list_user_policies").paginate(UserName=username):
        for policy_name in page.get("PolicyNames", []):
            plan.do(
                f"Delete inline IAM user policy: {policy_name}",
                iam.delete_user_policy,
                UserName=username,
                PolicyName=policy_name,
            )

    for page in iam.get_paginator("list_groups_for_user").paginate(UserName=username):
        for group in page.get("Groups", []):
            plan.do(
                f"Remove user from IAM group: {group['GroupName']}",
                iam.remove_user_from_group,
                UserName=username,
                GroupName=group["GroupName"],
            )

    try:
        plan.do(
            f"IAM login profile for user: {username}",
            iam.delete_login_profile,
            UserName=username,
        )
    except ClientError as exc:
        if not _ignore_not_found(exc, {"NoSuchEntity"}):
            raise

    for page in iam.get_paginator("list_mfa_devices").paginate(UserName=username):
        for device in page.get("MFADevices", []):
            plan.do(
                f"Deactivate MFA device: {device['SerialNumber']}",
                iam.deactivate_mfa_device,
                UserName=username,
                SerialNumber=device["SerialNumber"],
            )

    for page in iam.get_paginator("list_signing_certificates").paginate(UserName=username):
        for cert in page.get("Certificates", []):
            plan.do(
                f"Delete signing certificate: {cert['CertificateId']}",
                iam.delete_signing_certificate,
                UserName=username,
                CertificateId=cert["CertificateId"],
            )

    for page in iam.get_paginator("list_ssh_public_keys").paginate(UserName=username):
        for key in page.get("SSHPublicKeys", []):
            plan.do(
                f"Delete SSH public key: {key['SSHPublicKeyId']}",
                iam.delete_ssh_public_key,
                UserName=username,
                SSHPublicKeyId=key["SSHPublicKeyId"],
            )

    for cred in iam.list_service_specific_credentials(UserName=username).get(
        "ServiceSpecificCredentials", []
    ):
        plan.do(
            f"Delete service-specific credential: {cred['ServiceSpecificCredentialId']}",
            iam.delete_service_specific_credential,
            UserName=username,
            ServiceSpecificCredentialId=cred["ServiceSpecificCredentialId"],
        )

    plan.do(f"IAM user: {username}", iam.delete_user, UserName=username)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username", help="Premium customer IAM username to delete")
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region for Athena/S3/Secrets Manager (default: {DEFAULT_REGION})",
    )
    parser.add_argument("--profile", help="AWS profile name to use")
    parser.add_argument(
        "--results-bucket",
        help=(
            "Athena results bucket. Defaults to ATHENA_RESULTS_BUCKET in the "
            f"stack secret, falling back to {DEFAULT_ATHENA_RESULTS_BUCKET}."
        ),
    )
    parser.add_argument(
        "--secret-name",
        default=DEFAULT_SECRET_NAME,
        help=f"Secret to read defaults from (default: {DEFAULT_SECRET_NAME})",
    )
    parser.add_argument(
        "--skip-secret",
        action="store_true",
        help="Do not read Secrets Manager for defaults.",
    )
    parser.add_argument(
        "--skip-s3",
        action="store_true",
        help="Do not delete S3 Athena result objects.",
    )
    parser.add_argument(
        "--skip-athena",
        action="store_true",
        help="Do not delete the Athena workgroup.",
    )
    parser.add_argument(
        "--skip-iam",
        action="store_true",
        help="Do not delete IAM access keys/policies/user.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"Allow deleting a username not starting with {PREMIUM_CUSTOMER_USER_PREFIX}.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete resources. Without this flag, only prints a dry run.",
    )
    args = parser.parse_args()

    if not args.force and not args.username.startswith(PREMIUM_CUSTOMER_USER_PREFIX):
        parser.error(
            f"username must start with {PREMIUM_CUSTOMER_USER_PREFIX}; use --force to override"
        )

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    secret_config: dict[str, str] = {}
    if not args.skip_secret:
        secret_config = _load_secret_config(session, args.secret_name)

    results_bucket = (
        args.results_bucket
        or secret_config.get("ATHENA_RESULTS_BUCKET")
        or DEFAULT_ATHENA_RESULTS_BUCKET
    )
    results_prefix = f"{PREMIUM_CUSTOMER_RESULTS_PREFIX}{args.username}/"
    plan = DeletionPlan(dry_run=not args.yes)

    print(f"Username:          {args.username}")
    print(f"Athena workgroup:  {args.username}")
    print(f"S3 results prefix: s3://{results_bucket}/{results_prefix}")
    print(f"Mode:              {'DELETE' if args.yes else 'DRY RUN'}")
    print()

    if not args.skip_athena:
        _delete_athena_workgroup(plan, session, args.username)
    if not args.skip_s3:
        _delete_s3_prefix(plan, session, results_bucket, results_prefix)
    if not args.skip_iam:
        _delete_iam_user(plan, session, args.username)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())