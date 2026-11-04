"""Source records are transient; adapters own namespace and cursor semantics."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Protocol

@dataclass(frozen=True)
class Record:
    ns: str
    id: str
    time: str
    speaker: str
    text: str

class SourceAdapter(Protocol):
    def iter_since(self, cursor: str | None) -> Iterable[Record]: ...
    def get_by_ids(self, ids: Iterable[str]) -> dict[str, Record]: ...

class FixtureAdapter:
    """Exclusive fixture-order cursor (record ID), missing IDs are omitted.

    Unknown cursor raises rather than silently skipping a hole. Source errors
    propagate; callers may retry individual batches. Never writes source text.
    """
    def __init__(self, path: str | Path, ns: str = "fixture"):
        self.ns = ns
        self.records = [Record(**json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
        if any(r.ns != ns for r in self.records):
            raise ValueError("fixture namespace mismatch")
        if len({r.id for r in self.records}) != len(self.records):
            raise ValueError("duplicate source ID")

    def iter_since(self, cursor=None):
        start = 0
        if cursor is not None:
            start = [r.id for r in self.records].index(cursor) + 1
        yield from self.records[start:]

    def get_by_ids(self, ids):
        wanted = set(ids)
        return {r.id: r for r in self.records if r.id in wanted}
