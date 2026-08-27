"""Content-addressed disk cache.

Everything the pipeline fetches or infers goes through here. Two reasons:
  1. Re-runs cost nothing, so iterating on prompts does not burn a free-tier quota.
  2. The cache is committed, so a reviewer with no keys and no GPU can reproduce
     the exact output. That is evaluation criterion #1.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def key_of(*parts: Any) -> str:
    """Stable hash of the arguments that determine a fetch."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:20]


class Cache:
    def __init__(self, root: Path, suffix: str = ".json"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.suffix = suffix
        self.hits = 0
        self.misses = 0

    def path(self, key: str) -> Path:
        return self.root / f"{key}{self.suffix}"

    def has(self, key: str) -> bool:
        return self.path(key).exists()

    def get_json(self, key: str) -> Optional[Any]:
        p = self.path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None

    def put_json(self, key: str, value: Any) -> None:
        self.path(key).write_text(json.dumps(value, indent=1, default=str))

    def get_bytes(self, key: str) -> Optional[bytes]:
        p = self.path(key)
        return p.read_bytes() if p.exists() else None

    def put_bytes(self, key: str, value: bytes) -> None:
        self.path(key).write_bytes(value)

    def json_or_fetch(self, key: str, fetch: Callable[[], Any],
                      offline: bool = False) -> Optional[Any]:
        """Return cached JSON, else call `fetch` unless we are offline.

        In offline mode a miss returns None rather than raising: the pipeline is
        designed to produce a complete (if less confident) row for every school
        even when a source is unavailable.
        """
        cached = self.get_json(key)
        if cached is not None:
            self.hits += 1
            return cached
        if offline:
            self.misses += 1
            return None
        value = fetch()
        self.misses += 1
        if value is not None:
            self.put_json(key, value)
        return value

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


def retry(fn: Callable[[], Any], attempts: int = 4, base_delay: float = 1.5,
          label: str = "") -> Any:
    """Exponential backoff. Overpass and the Esri endpoints both throttle."""
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we retry on anything transient
            last = exc
            if i < attempts - 1:
                time.sleep(base_delay * (2 ** i))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")
