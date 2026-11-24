-- Runs once, the first time the Postgres volume is created.
-- POSTGRES_USER (docflow_owner) owns the tables and runs migrations.
-- docflow_app is what the API and workers log in as: not an owner, no BYPASSRLS,
-- so row-level security always applies to it. Local-only passwords.
CREATE ROLE docflow_app LOGIN PASSWORD 'app_dev_pw' NOBYPASSRLS;

-- A separate database for the test suite, so tests never touch your dev data.
CREATE DATABASE docflow_test OWNER docflow_owner;
