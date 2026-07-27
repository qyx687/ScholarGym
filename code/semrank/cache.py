"""Thread-safe persistent caches for SemRank profiles and concept vectors."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .models import PaperConceptProfile, QueryConceptProfile


SCHEMA_VERSION = "semrank_sqlite_v2_text_vector_split"


class SemRankCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._stats: Counter[str] = Counter()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=60,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=60000")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS paper_profiles (
                    cache_key TEXT PRIMARY KEY,
                    paper_arxiv_id TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS paper_profiles_arxiv
                    ON paper_profiles(paper_arxiv_id);
                CREATE TABLE IF NOT EXISTS query_profiles (
                    cache_key TEXT PRIMARY KEY,
                    query_id TEXT NOT NULL,
                    identity_json TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS query_profiles_qid
                    ON query_profiles(query_id);
                CREATE TABLE IF NOT EXISTS concept_embeddings (
                    cache_key TEXT PRIMARY KEY,
                    encoder_id TEXT NOT NULL,
                    concept_normalized TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_f32 BLOB NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS concept_embeddings_encoder
                    ON concept_embeddings(encoder_id);
                """
            )
            existing = self._connection.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()
            if existing and existing[0] != SCHEMA_VERSION:
                raise ValueError(
                    "SemRank cache schema mismatch: "
                    f"{existing[0]!r} != {SCHEMA_VERSION!r}"
                )
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES(?, ?)",
                ("schema_version", SCHEMA_VERSION),
            )

    @staticmethod
    def _json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def get_paper_profile(
        self, cache_key: str
    ) -> Optional[PaperConceptProfile]:
        with self._lock:
            row = self._connection.execute(
                "SELECT profile_json FROM paper_profiles WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            self._stats["paper_profile_misses"] += 1
            return None
        self._stats["paper_profile_hits"] += 1
        return PaperConceptProfile.from_dict(json.loads(row[0])).with_cache_hit(
            True
        )

    def put_paper_profile(
        self,
        cache_key: str,
        identity: Mapping[str, Any],
        profile: PaperConceptProfile,
    ) -> None:
        stored = profile.with_cache_hit(False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO paper_profiles(
                    cache_key, paper_arxiv_id, identity_json, profile_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    stored.paper_arxiv_id,
                    self._json(identity),
                    self._json(stored.to_dict(include_raw=True)),
                    time.time(),
                ),
            )
        self._stats["paper_profile_writes"] += 1

    def get_query_profile(
        self, cache_key: str
    ) -> Optional[QueryConceptProfile]:
        with self._lock:
            row = self._connection.execute(
                "SELECT profile_json FROM query_profiles WHERE cache_key=?",
                (cache_key,),
            ).fetchone()
        if row is None:
            self._stats["query_profile_misses"] += 1
            return None
        self._stats["query_profile_hits"] += 1
        return QueryConceptProfile.from_dict(json.loads(row[0])).with_cache_hit(
            True
        )

    def put_query_profile(
        self,
        cache_key: str,
        identity: Mapping[str, Any],
        profile: QueryConceptProfile,
    ) -> None:
        stored = profile.with_cache_hit(False)
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO query_profiles(
                    cache_key, query_id, identity_json, profile_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    stored.query_id,
                    self._json(identity),
                    self._json(stored.to_dict(include_raw=True)),
                    time.time(),
                ),
            )
        self._stats["query_profile_writes"] += 1

    def get_embedding(
        self, cache_key: str, *, encoder_id: str
    ) -> Optional[np.ndarray]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT dimension, vector_f32, encoder_id
                FROM concept_embeddings
                WHERE cache_key=?
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            self._stats["concept_embedding_misses"] += 1
            return None
        dimension, blob, stored_encoder = int(row[0]), row[1], str(row[2])
        if stored_encoder != encoder_id:
            self._stats["concept_embedding_identity_rejections"] += 1
            return None
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.size != dimension:
            raise ValueError(
                f"corrupt SemRank embedding cache row: expected "
                f"{dimension}, found {vector.size}"
            )
        self._stats["concept_embedding_hits"] += 1
        return vector

    def put_embedding(
        self,
        cache_key: str,
        *,
        encoder_id: str,
        concept_normalized: str,
        vector: np.ndarray,
    ) -> None:
        self.put_embeddings(
            [
                (
                    cache_key,
                    encoder_id,
                    concept_normalized,
                    vector,
                )
            ]
        )

    def put_embeddings(
        self,
        records: Iterable[Tuple[str, str, str, np.ndarray]],
    ) -> None:
        """Write one encoder batch in a single atomic SQLite transaction."""

        rows = []
        created_at = time.time()
        for cache_key, encoder_id, concept_normalized, vector in records:
            value = np.asarray(vector, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    str(cache_key),
                    str(encoder_id),
                    str(concept_normalized),
                    int(value.size),
                    sqlite3.Binary(value.tobytes(order="C")),
                    created_at,
                )
            )
        if not rows:
            return
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO concept_embeddings(
                    cache_key, encoder_id, concept_normalized, dimension,
                    vector_f32, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self._stats["concept_embedding_writes"] += len(rows)
        self._stats["concept_embedding_write_transactions"] += 1

    def table_counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                table: int(
                    self._connection.execute(
                        f"SELECT COUNT(*) FROM {table}"  # noqa: S608
                    ).fetchone()[0]
                )
                for table in (
                    "paper_profiles",
                    "query_profiles",
                    "concept_embeddings",
                )
            }

    def paper_profile_status_counts(self) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        with self._lock:
            rows = self._connection.execute(
                "SELECT profile_json FROM paper_profiles"
            ).fetchall()
        for (payload,) in rows:
            status = str(json.loads(payload).get("status") or "unknown")
            counts[status] += 1
        return {key: int(value) for key, value in sorted(counts.items())}

    def snapshot_stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SemRankCache":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
