import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class SessionStore:
    def __init__(self, storage_path: str | None = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._lock = Lock()
        self._sessions: dict[str, dict[str, Any]] = {}
        if self.storage_path:
            self._load()

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                self._sessions = payload
        except (json.JSONDecodeError, OSError):
            self._sessions = {}

    def _save(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(self._sessions, handle, indent=2, ensure_ascii=False)

    def get_or_create(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.setdefault(
                session_id,
                {
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "history": [],
                    "metadata": {},
                },
            )
            session["updated_at"] = datetime.now(timezone.utc).isoformat()
            return deepcopy(session)

    def append_turn(self, session_id: str, role: str, content: str, metadata: dict | None = None) -> dict[str, Any]:
        with self._lock:
            session = self.get_or_create(session_id)
            session["history"].append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": metadata or {},
                }
            )
            if len(session["history"]) > 50:
                session["history"] = session["history"][-50:]
            self._sessions[session_id] = session
            self._save()
            return deepcopy(session)

    def get_history(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id, {"history": []})
            history = list(session.get("history", []))
            if limit is not None:
                return history[-limit:]
            return history

    def update_metadata(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        with self._lock:
            session = self.get_or_create(session_id)
            session["metadata"].update(kwargs)
            session["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._sessions[session_id] = session
            self._save()
            return deepcopy(session)


session_store = SessionStore(storage_path=None)
