"""
PostGIS connection pool using psycopg2.
Provides a simple interface to get/release connections from a shared pool.
"""

import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool as pg_pool

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

# Module-level connection pool — initialized lazily
_connection_pool: pg_pool.SimpleConnectionPool | None = None


def _init_pool() -> pg_pool.SimpleConnectionPool:
    """Create the connection pool on first use."""
    settings = get_settings()
    try:
        return pg_pool.SimpleConnectionPool(
            minconn=2,
            maxconn=10,
            dsn=settings.postgres_dsn,
        )
    except psycopg2.OperationalError as exc:
        logger.error("Failed to connect to PostGIS: %s", exc)
        raise


def get_pool() -> pg_pool.SimpleConnectionPool:
    """Return the global pool, creating it if needed."""
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        _connection_pool = _init_pool()
    return _connection_pool


@contextmanager
def get_db_connection():
    """
    Context manager that checks out a connection from the pool
    and returns it when done. Auto-rolls-back on exception.

    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def close_pool() -> None:
    """Shut down the connection pool (call on app shutdown)."""
    global _connection_pool
    if _connection_pool is not None and not _connection_pool.closed:
        _connection_pool.closeall()
        logger.info("PostGIS connection pool closed.")
        _connection_pool = None
