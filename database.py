"""
database.py — SQLite persistence layer.
Handles schema creation, document insertion, and retrieval.
"""

import io
import os
import sqlite3

import numpy as np

DB_PATH = os.path.join(os.path.dirname(__file__), "documents.db")


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # Allows dict-style column access
    return conn


# ─────────────────────────────────────────────
# SCHEMA
# ─────────────────────────────────────────────

def init_db():
    """
    Create the documents table if it does not already exist.

    Schema:
        id            — auto-increment primary key
        path          — absolute file path (unique)
        text          — extracted cleaned text
        hash          — SHA-256 of file bytes (for exact dupe detection)
        embedding     — serialized numpy array (BLOB)
        file_size     — size in bytes
        modified_time — Unix epoch float from os.stat
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            path          TEXT    UNIQUE NOT NULL,
            text          TEXT    NOT NULL,
            hash          TEXT    NOT NULL,
            embedding     BLOB    NOT NULL,
            file_size     INTEGER,
            modified_time REAL
        )
    """)
    # Index on hash for fast duplicate lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hash ON documents (hash)
    """)
    conn.commit()
    conn.close()
    print("[DB] Database initialized.")


# ─────────────────────────────────────────────
# SERIALIZATION HELPERS
# ─────────────────────────────────────────────

def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Convert a numpy array to raw bytes for SQLite BLOB storage."""
    buf = io.BytesIO()
    np.save(buf, embedding)
    return buf.getvalue()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Reconstruct a numpy array from SQLite BLOB bytes."""
    buf = io.BytesIO(blob)
    return np.load(buf)


# ─────────────────────────────────────────────
# WRITE OPERATIONS
# ─────────────────────────────────────────────

def insert_document(path: str,
                    text: str,
                    file_hash: str,
                    embedding: np.ndarray,
                    file_size: int = None,
                    modified_time: float = None) -> None:
    """
    Insert or replace a document record in the database.

    Uses INSERT OR REPLACE so re-indexing an updated file
    (same path, new hash) overwrites the old record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO documents
            (path, text, hash, embedding, file_size, modified_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        path,
        text,
        file_hash,
        serialize_embedding(embedding),
        file_size,
        modified_time,
    ))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# READ OPERATIONS
# ─────────────────────────────────────────────

def check_existing_hash(file_hash: str) -> str | None:
    """
    Check whether a document with this SHA-256 hash already exists.

    Returns:
        The stored file path if found, or None.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT path FROM documents WHERE hash = ?", (file_hash,))
    row = cursor.fetchone()
    conn.close()
    return row["path"] if row else None


def get_all_documents() -> list[dict]:
    """
    Retrieve all documents with their deserialized embeddings.

    Returns:
        List of dicts with keys: id, path, text, hash, embedding,
        file_size, modified_time.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, path, text, hash, embedding, file_size, modified_time
        FROM documents
    """)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id":            row["id"],
            "path":          row["path"],
            "text":          row["text"],
            "hash":          row["hash"],
            "embedding":     deserialize_embedding(row["embedding"]),
            "file_size":     row["file_size"],
            "modified_time": row["modified_time"],
        })
    return results


def get_document_count() -> int:
    """Return the total number of indexed documents."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM documents")
    row = cursor.fetchone()
    conn.close()
    return row["cnt"]


def document_exists_by_path(path: str) -> bool:
    """Check if a document path is already indexed."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM documents WHERE path = ?", (path,))
    row = cursor.fetchone()
    conn.close()
    return row is not None