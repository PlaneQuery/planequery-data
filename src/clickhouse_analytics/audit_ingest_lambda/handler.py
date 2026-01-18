import os
import clickhouse_connect

INGEST_SQL = """
INSERT INTO default.audit_query_log
SELECT
    event_time,
    query_id,
    user,
    toString(client_address) AS client_address,
    client_hostname,
    query,
    query_duration_ms,
    read_rows,
    read_bytes,
    written_rows,
    written_bytes,
    result_rows,
    result_bytes,
    memory_usage,
    exception_code,
    exception
FROM system.query_log
WHERE type = 'QueryFinish'
  AND event_time > coalesce((SELECT max(event_time) FROM default.audit_query_log), toDateTime(0))
  AND event_time <= now() - INTERVAL 10 SECOND;
"""

def handler(event, context):
    client = clickhouse_connect.get_client(
        host=os.environ["CLICKHOUSE_HOST"],
        username="default",
        password=os.environ["CLICKHOUSE_PASSWORD"],
        secure=True,
    )
    client.command(INGEST_SQL)
    return {"ok": True}
