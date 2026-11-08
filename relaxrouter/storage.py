"""Small SQLite metadata store; each operation owns its connection."""
import json
import sqlite3
import threading
from pathlib import Path


class Store:
    def __init__(self, root: Path):
        self.root = root
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created TEXT, payload TEXT)")

    def connect(self):
        return sqlite3.connect(self.root / "jobs.sqlite3", timeout=30)

    def save(self, job: dict) -> None:
        with self.lock, self.connect() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)",
                       (job["id"], job["created"], json.dumps(job, allow_nan=False)))

    def get(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM jobs WHERE id=?", (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self) -> list[dict]:
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM jobs ORDER BY created DESC")]

    def directory(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id
