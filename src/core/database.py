"""SQLite database layer with FTS5 for lexical search."""

import sqlite3
import hashlib
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from contextlib import contextmanager
import asyncio


@dataclass
class Screenshot:
    """Represents a screenshot record in the database."""
    id: Optional[int]
    file_path: str
    file_hash: str
    app_name: Optional[str]
    window_title: Optional[str]
    captured_at: Optional[datetime]
    indexed_at: datetime
    ocr_text: Optional[str]
    visual_description: Optional[str]
    
    @classmethod
    def from_row(cls, row: tuple) -> "Screenshot":
        """Create a Screenshot from a database row."""
        return cls(
            id=row[0],
            file_path=row[1],
            file_hash=row[2],
            app_name=row[3],
            window_title=row[4],
            captured_at=datetime.fromisoformat(row[5]) if row[5] else None,
            indexed_at=datetime.fromisoformat(row[6]),
            ocr_text=row[7],
            visual_description=row[8],
        )


class Database:
    """SQLite database with FTS5 for screenshot indexing and search."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._connection() as conn:
            # Main screenshots table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    app_name TEXT,
                    window_title TEXT,
                    captured_at TEXT,
                    indexed_at TEXT NOT NULL,
                    ocr_text TEXT,
                    visual_description TEXT
                )
            """)
            
            # FTS5 virtual table for full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS screenshots_fts USING fts5(
                    ocr_text,
                    visual_description,
                    app_name,
                    window_title,
                    content='screenshots',
                    content_rowid='id'
                )
            """)
            
            # Triggers to keep FTS5 in sync with main table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS screenshots_ai AFTER INSERT ON screenshots BEGIN
                    INSERT INTO screenshots_fts(rowid, ocr_text, visual_description, app_name, window_title)
                    VALUES (new.id, new.ocr_text, new.visual_description, new.app_name, new.window_title);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS screenshots_ad AFTER DELETE ON screenshots BEGIN
                    INSERT INTO screenshots_fts(screenshots_fts, rowid, ocr_text, visual_description, app_name, window_title)
                    VALUES ('delete', old.id, old.ocr_text, old.visual_description, old.app_name, old.window_title);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS screenshots_au AFTER UPDATE ON screenshots BEGIN
                    INSERT INTO screenshots_fts(screenshots_fts, rowid, ocr_text, visual_description, app_name, window_title)
                    VALUES ('delete', old.id, old.ocr_text, old.visual_description, old.app_name, old.window_title);
                    INSERT INTO screenshots_fts(rowid, ocr_text, visual_description, app_name, window_title)
                    VALUES (new.id, new.ocr_text, new.visual_description, new.app_name, new.window_title);
                END
            """)
            
            # Index on file_hash for quick change detection
            conn.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON screenshots(file_hash)")
            
            conn.commit()
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def get_by_path(self, file_path: str) -> Optional[Screenshot]:
        """Get a screenshot by its file path."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM screenshots WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()
            return Screenshot.from_row(row) if row else None
    
    def get_by_id(self, screenshot_id: int) -> Optional[Screenshot]:
        """Get a screenshot by its ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM screenshots WHERE id = ?",
                (screenshot_id,)
            )
            row = cursor.fetchone()
            return Screenshot.from_row(row) if row else None
    
    def insert(self, screenshot: Screenshot) -> int:
        """Insert a new screenshot, returning the new ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO screenshots 
                (file_path, file_hash, app_name, window_title, captured_at, indexed_at, ocr_text, visual_description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    screenshot.file_path,
                    screenshot.file_hash,
                    screenshot.app_name,
                    screenshot.window_title,
                    screenshot.captured_at.isoformat() if screenshot.captured_at else None,
                    screenshot.indexed_at.isoformat(),
                    screenshot.ocr_text,
                    screenshot.visual_description,
                )
            )
            conn.commit()
            return cursor.lastrowid
    
    def update(self, screenshot: Screenshot) -> None:
        """Update an existing screenshot."""
        if screenshot.id is None:
            raise ValueError("Cannot update screenshot without ID")
        
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE screenshots SET
                    file_path = ?,
                    file_hash = ?,
                    app_name = ?,
                    window_title = ?,
                    captured_at = ?,
                    indexed_at = ?,
                    ocr_text = ?,
                    visual_description = ?
                WHERE id = ?
                """,
                (
                    screenshot.file_path,
                    screenshot.file_hash,
                    screenshot.app_name,
                    screenshot.window_title,
                    screenshot.captured_at.isoformat() if screenshot.captured_at else None,
                    screenshot.indexed_at.isoformat(),
                    screenshot.ocr_text,
                    screenshot.visual_description,
                    screenshot.id,
                )
            )
            conn.commit()
    
    def delete(self, screenshot_id: int) -> None:
        """Delete a screenshot by ID."""
        with self._connection() as conn:
            conn.execute("DELETE FROM screenshots WHERE id = ?", (screenshot_id,))
            conn.commit()
    
    def get_all_paths_and_hashes(self) -> dict[str, str]:
        """Get all indexed file paths and their hashes for change detection."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT file_path, file_hash FROM screenshots")
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def get_by_paths_batch(self, file_paths: list[str]) -> dict[str, Optional[Screenshot]]:
        """Get multiple screenshots by their file paths in a single query.
        
        Args:
            file_paths: List of file paths to look up
            
        Returns:
            Dictionary mapping file_path -> Screenshot (or None if not found)
        """
        if not file_paths:
            return {}
        
        with self._connection() as conn:
            # Use IN clause for batch lookup
            placeholders = ','.join('?' * len(file_paths))
            cursor = conn.execute(
                f"SELECT * FROM screenshots WHERE file_path IN ({placeholders})",
                file_paths
            )
            results = {row[1]: Screenshot.from_row(row) for row in cursor.fetchall()}
            # Add None for paths not found
            return {path: results.get(path) for path in file_paths}
    
    # Async wrappers for non-blocking operations
    async def get_by_path_async(self, file_path: str) -> Optional[Screenshot]:
        """Async wrapper for get_by_path."""
        return await asyncio.to_thread(self.get_by_path, file_path)
    
    async def get_by_paths_batch_async(self, file_paths: list[str]) -> dict[str, Optional[Screenshot]]:
        """Async wrapper for get_by_paths_batch."""
        return await asyncio.to_thread(self.get_by_paths_batch, file_paths)
    
    async def insert_async(self, screenshot: Screenshot) -> int:
        """Async wrapper for insert."""
        return await asyncio.to_thread(self.insert, screenshot)
    
    async def update_async(self, screenshot: Screenshot) -> None:
        """Async wrapper for update."""
        return await asyncio.to_thread(self.update, screenshot)
    
    def fts_search(self, query: str, limit: int = 50) -> list[Screenshot]:
        """Search using FTS5 full-text search."""
        with self._connection() as conn:
            # Use FTS5 MATCH syntax
            cursor = conn.execute(
                """
                SELECT screenshots.* FROM screenshots
                JOIN screenshots_fts ON screenshots.id = screenshots_fts.rowid
                WHERE screenshots_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit)
            )
            return [Screenshot.from_row(row) for row in cursor.fetchall()]
    
    def get_count(self) -> int:
        """Get total number of indexed screenshots."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM screenshots")
            return cursor.fetchone()[0]
    
    def get_count_by_folder(self, folder_path: str) -> int:
        """Get count of screenshots indexed from a specific folder."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM screenshots WHERE file_path LIKE ?",
                (f"{folder_path}%",)
            )
            return cursor.fetchone()[0]


def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file for change detection."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
