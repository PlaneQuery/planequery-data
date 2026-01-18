CREATE TABLE audit_query_log
AS system.query_log
ENGINE = MergeTree
ORDER BY event_time;
