# cdk/cdk/swim_realtime_stack.py
from aws_cdk import (
    Stack,
    CfnOutput,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct


class SwimRealtimeStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1) VPC: use default VPC
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # 2) Security group: allow SSH inbound
        sg = ec2.SecurityGroup(
            self,
            "SwimDevSg",
            vpc=vpc,
            description="SWIM Dev EC2 Security Group",
            allow_all_outbound=True,
        )
        sg.add_ingress_rule(ec2.Peer.any_ipv4(), ec2.Port.tcp(22), "SSH")

        # 3) Instance role: SSM for management
        role = iam.Role(
            self,
            "SwimDevInstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
        )
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        # Grant S3 write access to planequery-swim-logs-prod bucket
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:PutObject",
                    "s3:PutObjectAcl",
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    "arn:aws:s3:::planequery-swim-logs-prod",
                    "arn:aws:s3:::planequery-swim-logs-prod/*",
                ],
            )
        )

        # 4) Ubuntu 24.04 LTS ARM64 AMI for us-east-2
        ubuntu_ami = ec2.MachineImage.generic_linux(
            {
                "us-east-2": "ami-01da1dbf9ea3a6ee6",
            }
        )

        # 5) EC2 instance: t4g.micro, Ubuntu ARM64
        instance = ec2.Instance(
            self,
            "SwimDevInstance",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_group=sg,
            role=role,
            instance_type=ec2.InstanceType("t4g.medium"),
            machine_image=ubuntu_ami,
            key_name="planequery-dev",
        )

        # 6) Basic UserData: update packages
        instance.add_user_data(
            "#!/bin/bash",
            "set -euxo pipefail",
            "export DEBIAN_FRONTEND=noninteractive",
            "apt-get update -y",
            "apt-get upgrade -y",
        )

        # 7) Allocate and attach an Elastic IP
        eip = ec2.CfnEIP(self, "SwimDevEip", domain="vpc")

        ec2.CfnEIPAssociation(
            self,
            "SwimDevEipAssoc",
            allocation_id=eip.attr_allocation_id,
            instance_id=instance.instance_id,
        )

        # Outputs
        CfnOutput(self, "PublicIp", value=eip.attr_public_ip)
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(
            self,
            "SSHCommand",
            value=f"ssh -i ~/.ssh/planequery-dev.pem ubuntu@{eip.attr_public_ip}",
        )