from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("POSTGRES_URL")
    or os.environ.get("SUPABASE_DB_URL")
)
USE_POSTGRES = bool(DATABASE_URL)


class HybridRow(dict):
    """Dict-like row that also supports row[0] like sqlite3.Row."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def _pg_row_factory(cursor):
    if cursor.description is None:
        return lambda values: values
    names = [col.name for col in cursor.description]
    return lambda values: HybridRow(zip(names, values))


class PostgresConnection:
    def __init__(self, url: str):
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("PostgreSQL support requires psycopg. Install requirements.txt.") from exc
        self._conn = psycopg.connect(
            url,
            row_factory=_pg_row_factory,
            connect_timeout=10,
            prepare_threshold=None,
        ) 

    def execute(self, sql, params=()):
        sql = sql.replace("datetime('now','localtime')", "CURRENT_TIMESTAMP")
        sql = sql.replace("?", "%s")
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()


def db(root: Path):
    if USE_POSTGRES:
        return PostgresConnection(DATABASE_URL)
    data = root / "data"
    data.mkdir(exist_ok=True)
    con = sqlite3.connect(data / "threatlens.db", timeout=10)
    con.row_factory = sqlite3.Row
    return con


def is_integrity_error(exc: Exception) -> bool:
    if isinstance(exc, sqlite3.IntegrityError):
        return True
    return getattr(exc, "sqlstate", None) == "23505"


def init_db(root: Path):
    con = db(root)
    if USE_POSTGRES:
        statements = [
            """CREATE TABLE IF NOT EXISTS users(
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
            )""",
            """CREATE TABLE IF NOT EXISTS scans(
              id BIGSERIAL PRIMARY KEY,
              user_id BIGINT NOT NULL REFERENCES users(id),
              url TEXT NOT NULL,
              verdict TEXT NOT NULL,
              risk DOUBLE PRECISION NOT NULL,
              confidence DOUBLE PRECISION NOT NULL,
              model TEXT NOT NULL,
              features_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS audit_log(
              id BIGSERIAL PRIMARY KEY,
              actor_user_id BIGINT REFERENCES users(id),
              event TEXT NOT NULL,
              target_user_id BIGINT REFERENCES users(id),
              detail TEXT,
              ip TEXT,
              created_at TEXT NOT NULL
            )""",
        ]
        try:
            for statement in statements:
                con.execute(statement)
            con.commit()
        finally:
            con.close()
        return

    try:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          email TEXT UNIQUE NOT NULL,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL DEFAULT 'Analyst',
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scans(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          url TEXT NOT NULL,
          verdict TEXT NOT NULL,
          risk REAL NOT NULL,
          confidence REAL NOT NULL,
          model TEXT NOT NULL,
          features_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS audit_log(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          actor_user_id INTEGER,
          event TEXT NOT NULL,
          target_user_id INTEGER,
          detail TEXT,
          ip TEXT,
          created_at TEXT NOT NULL,
          FOREIGN KEY(actor_user_id) REFERENCES users(id),
          FOREIGN KEY(target_user_id) REFERENCES users(id)
        );
        """)
        for col, definition in [
            ("status", "TEXT NOT NULL DEFAULT 'Active'"),
            ("failed_attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("locked_until", "TEXT"),
            ("last_login_at", "TEXT"),
            ("password_changed_at", "TEXT"),
            ("session_version", "INTEGER NOT NULL DEFAULT 1"),
            ("mfa_enabled", "INTEGER NOT NULL DEFAULT 0"),
        ]:
            try:
                con.execute(f"ALTER TABLE users ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass
        con.commit()
    finally:
        con.close()
