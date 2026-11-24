# Phase 1: Foundation

Phase 1 builds everything the pipeline will stand on: one codebase shared by the API and the
workers, a single configuration object, document schemas that are versioned and immutable,
value normalisation, and a database that enforces tenant isolation and auditability itself.

## 1. One workspace, three packages

`uv` workspace with `docflow-core`, `docflow-api` and `docflow-worker`. The API and the workers
both depend on core, so an invoice is extracted by exactly the same code whether it arrives
through an HTTP request or a queue message. One lockfile (`uv.lock`) pins every version for
the laptop, CI and the container image alike.

## 2. Configuration as environment variables

`docflow_core/config.py` is the only place settings are read. Locally they come from `.env`;
on AWS from SSM Parameter Store. The review threshold is deliberately read on every call
rather than cached, so it can be changed without a rebuild.

## 3. Schemas are the contract

`InvoiceV1` is a Pydantic model, and it does three jobs:

1. generates the JSON schema sent to the LLM (phase 3),
2. validates the model's reply,
3. defines the fields the review UI shows.

Schemas are registered as `(doc_type, version)`. A published version is immutable: the
registry compares a fingerprint (hash of the JSON schema) and refuses an edited
re-registration, and `publish_schemas` fails a deploy if code and database disagree.
`InvoiceV2` shows the rule in action: it adds `vendor_tax_id` and `po_number` (needed later
by the tax-ID validator and the purchase-order tool), and every v1 extraction stays valid.

## 4. Normalisation is shared

`"Rs. 1,180.00"`, `"₹1,180"` and `"INR 1,180.00"` are the same amount. `normalise.py` handles
currency tokens, Western and Indian digit grouping, the European decimal comma, accounting
negatives, day-first dates and company suffixes. The validators (phase 4) and the evals
(phase 5) both use it, so "correct" means the same thing in both places.

A bug found while building this: the pattern `rs\.?\b` matched only `Rs` in `Rs. 1,180.00`,
because there is no word boundary between a dot and a space. The test caught it.

## 5. The database enforces the rules

Postgres 16 with pgvector. The migration is plain SQL
(`db/migrations/sql/0001_initial.sql`), so the security rules can be read directly.

**Two roles.** `docflow_owner` owns the tables and runs migrations. The API and workers log in
as `docflow_app`, which owns nothing and has no `BYPASSRLS`. Row-level security therefore
always applies to application code.

**Row-level security.** Every tenant table has the policy
`tenant_id = current_tenant_id()`. `tenant_session()` sets the tenant with
`set_config('app.tenant_id', ..., true)`, which lasts only for the current transaction, so a
pooled connection cannot leak one tenant's id into the next request. `current_tenant_id()`
wraps the setting in `NULLIF(..., '')` because after a transaction ends the setting becomes an
empty string, and `''::uuid` would raise instead of returning no rows.

**API keys before the tenant is known.** Looking up a key happens before we know the tenant,
which RLS would block. A narrow `SECURITY DEFINER` function `auth_api_key(hash)` returns only
who the key belongs to. Only SHA-256 hashes of keys are stored.

**Append-only corrections.** Two layers: the app role is granted only `SELECT, INSERT`, and a
trigger rejects `UPDATE`, `DELETE` and `TRUNCATE` even for the owner. Corrections are both
the audit trail and the training data, so history must not change.

**Constraints that carry design rules.**

- `documents UNIQUE (tenant_id, sha256)`: re-uploading a file is free.
- `extractions UNIQUE (document_id, extractor_version_id)`: one extractor scores a document once.
- `extractor_versions` partial unique index on `doc_type WHERE status = 'active'`: two active
  extractors are impossible, even with a buggy promotion script.
- `stage_runs UNIQUE (document_id, stage)`: the idempotency key every worker checks (phase 2).
- `chunks`: HNSW index for cosine vector search and a GIN index on a generated `tsvector`
  column for keyword search (phase 6).

## 6. Tests that prove the guarantees

| Test | Proves |
|---|---|
| `test_a_tenant_sees_zero_rows_of_another_tenant` | RLS isolation, as the design requires |
| `test_a_tenant_cannot_write_rows_for_another_tenant` | `WITH CHECK` blocks cross-tenant writes |
| `test_no_tenant_set_means_no_rows` | Forgetting to set a tenant fails closed |
| `test_corrections_are_append_only` | Both the grant and the trigger |
| `test_only_one_active_extractor_per_doc_type` | The partial unique index |
| `test_models_match_the_migrated_schema` | Hand-written SQL and ORM models agree |

Database tests run against a separate `docflow_test` database, and skip cleanly when Postgres
is not running, so `pytest` always works.

A second bug the database tests caught: the first migration ran its SQL file through
SQLAlchemy's `exec_driver_sql`, which still passes psycopg an empty parameter list. psycopg
then reads the `%` in `format('%I', ...)` and `RAISE EXCEPTION '% is append-only'` as
placeholders, and fails. The migration now hands the file straight to the psycopg connection
with no parameters, inside the same Alembic transaction.

## Deviations from the design doc

- Added `stage_runs` (idempotency per stage) and `tenant_id` on child tables such as `pages`
  and `corrections`, so every RLS policy is a simple column check with no joins.
- Status columns are `text` with `CHECK` constraints instead of Postgres enums, so adding a
  status is a one-line migration.
