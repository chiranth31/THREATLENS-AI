-- Reference schema for PostgreSQL providers such as Supabase or Neon.
-- ThreatLens AI also creates this schema automatically at application startup.
CREATE TABLE IF NOT EXISTS users(
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'Analyst',
  created_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'Active',
  failed_attempts INTEGER NOT NULL DEFAULT 0,
  locked_until TEXT,
  last_login_at TEXT,
  password_changed_at TEXT,
  session_version INTEGER NOT NULL DEFAULT 1,
  mfa_enabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS scans(
  id BIGSERIAL PRIMARY KEY,
  user_id BIGINT NOT NULL REFERENCES users(id),
  url TEXT NOT NULL,
  verdict TEXT NOT NULL,
  risk DOUBLE PRECISION NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  model TEXT NOT NULL,
  features_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log(
  id BIGSERIAL PRIMARY KEY,
  actor_user_id BIGINT REFERENCES users(id),
  event TEXT NOT NULL,
  target_user_id BIGINT REFERENCES users(id),
  detail TEXT,
  ip TEXT,
  created_at TEXT NOT NULL
);
