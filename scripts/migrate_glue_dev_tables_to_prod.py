import boto3
import copy

REGION = "us-east-2"
OLD_DB = "planequery_lake_dev"
NEW_DB = "planequery_lake_prod"

TABLES = ["adsb_messages"]

glue = boto3.client("glue", region_name=REGION)

for table_name in TABLES:
    old = glue.get_table(DatabaseName=OLD_DB, Name=table_name)["Table"]

    new_table = {
        k: copy.deepcopy(old[k])
        for k in [
            "Name",
            "Description",
            "Owner",
            "LastAccessTime",
            "LastAnalyzedTime",
            "Retention",
            "StorageDescriptor",
            "PartitionKeys",
            "ViewOriginalText",
            "ViewExpandedText",
            "TableType",
            "Parameters",
            "TargetTable",
        ]
        if k in old
    }

    # Important: do NOT change StorageDescriptor.Location or Iceberg metadata params.
    glue.create_table(DatabaseName=NEW_DB, TableInput=new_table)
    print(f"Copied {OLD_DB}.{table_name} -> {NEW_DB}.{table_name}")