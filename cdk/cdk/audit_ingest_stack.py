import os
from aws_cdk import (
    Duration,
    Stack,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
)
from constructs import Construct

class AuditIngestStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # repo root/.env.prod (given your structure: repo/cdk/cdk/this_file.py)
        env_path = ".env.prod"

        env_vars = load_env_file(env_path)
        # remove blanks/unset
        env_vars = {k: v for k, v in env_vars.items() if v is not None}

        fn = _lambda.DockerImageFunction(
            self,
            "AuditIngestFn",
            code=_lambda.DockerImageCode.from_image_asset("../src/clickhouse_analytics/audit_ingest_lambda"),
            timeout=Duration.seconds(60),
            memory_size=256,
            environment=env_vars,
            architecture=_lambda.Architecture.ARM_64,  # Add this line
        )

        rule = events.Rule(
            self,
            "AuditIngestSchedule",
            schedule=events.Schedule.rate(Duration.minutes(10)),
        )
        rule.add_target(targets.LambdaFunction(fn))

def load_env_file(path: str) -> dict[str, str]:
    env = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env
