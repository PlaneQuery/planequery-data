#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cdk.cdk_stack import CdkStack
from cdk.swim_realtime_stack import SwimRealtimeStack
from cdk.planequery_adsb_realtime_stack import PlanequeryAdsbRealtimeStack
from cdk.planequery_datalake_stack import PlanequeryDatalakeStack
from cdk.planequery_datalake_adsb_insert_stack import PlanequeryDatalakeAdsbInsertStack
from cdk.planequery_daily_pipeline_stack import PlanequeryDailyPipelineStack
from cdk.planequery_inference_features_stack import PlanequeryInferenceFeaturesStack
from cdk.planequery_flight_inference_batch_stack import PlanequeryFlightInferenceBatchStack

app = cdk.App()
CdkStack(app, "CdkStack",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.

    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */

    #env=cdk.Environment(account='123456789012', region='us-east-1'),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
    )
SwimRealtimeStack(
    app,
    "SwimRealtimeStack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
PlanequeryAdsbRealtimeStack(
    app,
    "PlanequeryAdsbRealtimeStack",
    stack_name="planequery-adsb-realtime-stack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
PlanequeryDatalakeStack(
    app,
    "PlanequeryDatalakeStack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
adsb_insert_stack = PlanequeryDatalakeAdsbInsertStack(
    app,
    "PlanequeryDatalakeAdsbInsertStack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
inference_features_stack = PlanequeryInferenceFeaturesStack(
    app,
    "PlanequeryInferenceFeaturesStack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
flight_inference_stack = PlanequeryFlightInferenceBatchStack(
    app,
    "PlanequeryFlightInferenceBatchStack",
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
PlanequeryDailyPipelineStack(
    app,
    "PlanequeryDailyPipelineStack",
    adsb_insert_stack=adsb_insert_stack,
    inference_features_stack=inference_features_stack,
    flight_inference_stack=flight_inference_stack,
    env=cdk.Environment(
        account=os.getenv('CDK_DEFAULT_ACCOUNT'),
        region=os.getenv('CDK_DEFAULT_REGION')
    ),
)
app.synth()
