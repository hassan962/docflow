DROP TABLE IF EXISTS
    purchase_orders, vendors, chunks, usage_events, eval_runs, corrections, extractions,
    extractor_versions, schema_versions, stage_runs, pages, documents, api_keys, users, tenants
    CASCADE;
DROP FUNCTION IF EXISTS auth_api_key(text);
DROP FUNCTION IF EXISTS forbid_change();
DROP FUNCTION IF EXISTS set_updated_at();
DROP FUNCTION IF EXISTS current_tenant_id();
