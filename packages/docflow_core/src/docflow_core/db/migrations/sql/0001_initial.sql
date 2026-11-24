-- DocFlow initial schema.
--
-- Runs as the table owner (docflow_owner). The API and workers connect as docflow_app, which
-- does NOT own the tables and has no BYPASSRLS, so the row-level security policies below
-- always apply to it. Tenant isolation is enforced by the database, not only by API code.

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: embedding column + HNSW index
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- trigram similarity for fuzzy vendor lookup

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'docflow_app') THEN
        -- Login and password are set per environment (docker init script, or SSM in AWS).
        CREATE ROLE docflow_app NOLOGIN NOBYPASSRLS;
    END IF;
END
$$;

-- The tenant for the current transaction, or NULL when none is set. NULLIF matters: once a
-- connection has used SET LOCAL, the setting reverts to '' (not NULL) and ''::uuid would
-- raise instead of quietly matching no rows.
CREATE FUNCTION current_tenant_id() RETURNS uuid
    LANGUAGE sql STABLE
    AS $$ SELECT NULLIF(current_setting('app.tenant_id', true), '')::uuid $$;

CREATE FUNCTION set_updated_at() RETURNS trigger
    LANGUAGE plpgsql
    AS $$ BEGIN NEW.updated_at := now(); RETURN NEW; END $$;

-- ---------------------------------------------------------------- tenancy and access

CREATE TABLE tenants (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name             text NOT NULL,
    daily_budget_usd numeric(10, 4) NOT NULL DEFAULT 1.0,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  uuid NOT NULL REFERENCES tenants (id),
    email      text NOT NULL,
    role       text NOT NULL CHECK (role IN ('admin', 'reviewer', 'viewer')),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

CREATE TABLE api_keys (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  uuid NOT NULL REFERENCES tenants (id),
    user_id    uuid NOT NULL REFERENCES users (id),
    prefix     varchar(12) NOT NULL,
    key_hash   text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    revoked_at timestamptz
);

-- ---------------------------------------------------------------- documents and pipeline

CREATE TABLE documents (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      uuid NOT NULL REFERENCES tenants (id),
    s3_key         text NOT NULL,
    sha256         varchar(64) NOT NULL,
    filename       text NOT NULL,
    content_type   text NOT NULL,
    doc_type       text NOT NULL DEFAULT 'invoice',
    status         text NOT NULL DEFAULT 'uploaded' CHECK (status IN (
                       'uploaded', 'ocr_done', 'extracted', 'validated', 'needs_review',
                       'approved', 'indexed', 'failed')),
    page_count     integer,
    failure_reason text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, sha256)  -- re-uploading the same file is a no-op
);
CREATE INDEX documents_tenant_status_idx ON documents (tenant_id, status);
CREATE TRIGGER documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TABLE pages (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     uuid NOT NULL REFERENCES tenants (id),
    document_id   uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    page_no       integer NOT NULL CHECK (page_no >= 1),
    image_key     text NOT NULL,
    width         integer,
    height        integer,
    ocr_engine    text,
    ocr_json_key  text,
    mean_ocr_conf double precision,
    UNIQUE (document_id, page_no)
);

-- One row per (document, stage). A worker that finds status = 'done' acknowledges the message
-- and exits, which is what makes every stage safe to retry.
CREATE TABLE stage_runs (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   uuid NOT NULL REFERENCES tenants (id),
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    stage       text NOT NULL CHECK (stage IN ('ocr', 'extract', 'validate', 'index')),
    status      text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'done', 'failed')),
    attempts    integer NOT NULL DEFAULT 1,
    error       text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    UNIQUE (document_id, stage)
);

-- ---------------------------------------------------------------- schemas and extractors

CREATE TABLE schema_versions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type    text NOT NULL,
    version     integer NOT NULL CHECK (version >= 1),
    json_schema jsonb NOT NULL,
    fingerprint varchar(64) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (doc_type, version)
);

CREATE TABLE extractor_versions (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_type       text NOT NULL,
    schema_version integer NOT NULL,
    model          text NOT NULL,
    prompt_hash    varchar(64) NOT NULL,
    fewshot_policy text NOT NULL DEFAULT 'none',
    config         jsonb NOT NULL DEFAULT '{}'::jsonb,
    status         text NOT NULL DEFAULT 'candidate'
                       CHECK (status IN ('candidate', 'active', 'rejected', 'retired')),
    parent_id      uuid REFERENCES extractor_versions (id),
    created_at     timestamptz NOT NULL DEFAULT now(),
    promoted_at    timestamptz,
    FOREIGN KEY (doc_type, schema_version) REFERENCES schema_versions (doc_type, version)
);
-- At most one active extractor per document type, enforced by the database.
CREATE UNIQUE INDEX extractor_versions_one_active
    ON extractor_versions (doc_type) WHERE status = 'active';

CREATE TABLE extractions (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id            uuid NOT NULL REFERENCES tenants (id),
    document_id          uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    extractor_version_id uuid NOT NULL REFERENCES extractor_versions (id),
    fields               jsonb NOT NULL DEFAULT '{}'::jsonb,
    status               text NOT NULL DEFAULT 'ok'
                             CHECK (status IN ('ok', 'failed_parse', 'needs_review', 'approved')),
    revision             integer NOT NULL DEFAULT 1,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    UNIQUE (document_id, extractor_version_id)  -- one extractor scores a document once
);
CREATE TRIGGER extractions_updated_at BEFORE UPDATE ON extractions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Append-only: the audit trail and the training data. Nothing may rewrite history.
CREATE TABLE corrections (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id     uuid NOT NULL REFERENCES tenants (id),
    extraction_id uuid NOT NULL REFERENCES extractions (id),
    field         text NOT NULL,
    action        text NOT NULL CHECK (action IN ('accept', 'edit')),
    old_value     jsonb,
    new_value     jsonb,
    reviewer_id   uuid NOT NULL REFERENCES users (id),
    seconds_spent double precision,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX corrections_extraction_idx ON corrections (extraction_id);
CREATE INDEX corrections_field_time_idx ON corrections (field, created_at);

CREATE FUNCTION forbid_change() RETURNS trigger
    LANGUAGE plpgsql
    AS $$ BEGIN RAISE EXCEPTION '% is append-only', TG_TABLE_NAME; END $$;
CREATE TRIGGER corrections_append_only BEFORE UPDATE OR DELETE ON corrections
    FOR EACH ROW EXECUTE FUNCTION forbid_change();
CREATE TRIGGER corrections_no_truncate BEFORE TRUNCATE ON corrections
    FOR EACH STATEMENT EXECUTE FUNCTION forbid_change();

-- ---------------------------------------------------------------- evaluation and metering

CREATE TABLE eval_runs (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    extractor_version_id uuid NOT NULL REFERENCES extractor_versions (id),
    dataset              text NOT NULL,
    n_docs               integer NOT NULL,
    metrics              jsonb NOT NULL,
    cost_usd             numeric(12, 6) NOT NULL,
    p95_latency_ms       integer NOT NULL,
    report_key           text,
    created_at           timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE usage_events (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id   uuid NOT NULL REFERENCES tenants (id),
    document_id uuid REFERENCES documents (id) ON DELETE SET NULL,
    stage       text NOT NULL,
    model       text,
    tokens_in   integer NOT NULL DEFAULT 0,
    tokens_out  integer NOT NULL DEFAULT 0,
    cost_usd    numeric(12, 6) NOT NULL DEFAULT 0,
    latency_ms  integer NOT NULL DEFAULT 0,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX usage_events_tenant_time_idx ON usage_events (tenant_id, created_at);

-- ---------------------------------------------------------------- retrieval

CREATE TABLE chunks (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id   uuid NOT NULL REFERENCES tenants (id),
    document_id uuid NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    page_no     integer NOT NULL,
    kind        text NOT NULL DEFAULT 'text' CHECK (kind IN ('text', 'field')),
    text        text NOT NULL,
    bbox        jsonb,
    embedding   vector(384) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_gin ON chunks USING gin (tsv);
CREATE INDEX chunks_tenant_doc_idx ON chunks (tenant_id, document_id);

-- ---------------------------------------------------------------- reference data

CREATE TABLE vendors (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       uuid NOT NULL REFERENCES tenants (id),
    name            text NOT NULL,
    normalised_name text NOT NULL,
    tax_id          text
);
CREATE INDEX vendors_name_trgm ON vendors USING gin (normalised_name gin_trgm_ops);

CREATE TABLE purchase_orders (
    id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id uuid NOT NULL REFERENCES tenants (id),
    po_number text NOT NULL,
    vendor_id uuid NOT NULL REFERENCES vendors (id),
    total     numeric(14, 2) NOT NULL,
    currency  varchar(3) NOT NULL,
    status    text NOT NULL DEFAULT 'open',
    UNIQUE (tenant_id, po_number)
);

-- ---------------------------------------------------------------- row-level security

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenants
    USING (id = current_tenant_id()) WITH CHECK (id = current_tenant_id());

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'users', 'api_keys', 'documents', 'pages', 'stage_runs', 'extractions', 'corrections',
        'usage_events', 'chunks', 'vendors', 'purchase_orders'
    ] LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I '
            'USING (tenant_id = current_tenant_id()) '
            'WITH CHECK (tenant_id = current_tenant_id())', t);
    END LOOP;
END
$$;

-- API-key authentication has to find the key BEFORE the tenant is known, which RLS would
-- block. This narrow SECURITY DEFINER function is the only way around it: it takes a hash,
-- returns who it belongs to, and nothing else.
CREATE FUNCTION auth_api_key(p_key_hash text)
    RETURNS TABLE (tenant_id uuid, user_id uuid, role text)
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public
    AS $$
        SELECT k.tenant_id, k.user_id, u.role
        FROM api_keys k JOIN users u ON u.id = k.user_id
        WHERE k.key_hash = p_key_hash AND k.revoked_at IS NULL
    $$;
REVOKE ALL ON FUNCTION auth_api_key(text) FROM PUBLIC;

-- ---------------------------------------------------------------- grants for the app role

GRANT USAGE ON SCHEMA public TO docflow_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    tenants, users, api_keys, documents, pages, stage_runs, extractions, usage_events, chunks,
    vendors, purchase_orders
    TO docflow_app;
GRANT SELECT, INSERT ON corrections TO docflow_app;            -- never UPDATE/DELETE
GRANT SELECT, INSERT ON schema_versions TO docflow_app;
GRANT SELECT, INSERT, UPDATE ON extractor_versions TO docflow_app;
GRANT SELECT, INSERT ON eval_runs TO docflow_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO docflow_app;
GRANT EXECUTE ON FUNCTION auth_api_key(text), current_tenant_id() TO docflow_app;
