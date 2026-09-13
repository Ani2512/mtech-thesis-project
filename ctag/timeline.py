"""Event timelines and the predicates that define every condition type.

See docs/query_semantics.md. Everything here is pure and deterministic so the
ground truth of a query is a function of the timeline alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    label: str
    onset: float
    offset: float

    def __post_init__(self):
        if self.offset <= self.onset:
            raise ValueError(f"empty event {self}")

    @property
    def interval(self) -> tuple[float, float]:
        return (round(self.onset, 3), round(self.offset, 3))

    def overlap(self, other: "Event") -> float:
        return max(0.0, min(self.offset, other.offset) - max(self.onset, other.onset))


@dataclass
class Timeline:
    duration: float
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self.events = sorted(self.events, key=lambda e: (e.onset, e.offset))

    def labels(self) -> list[str]:
        seen: dict[str, None] = {}
        for e in self.events:
            seen.setdefault(e.label, None)
        return list(seen)

    def occ(self, label: str) -> list[Event]:
        return [e for e in self.events if e.label == label]

    # ------------------------------------------------------------ predicates
    def plain(self, x: str) -> list[Event]:
        return self.occ(x)

    def ordinal(self, x: str, k: int | str) -> list[Event]:
        xs = self.occ(x)
        if k == "last":
            return xs[-1:] if xs else []
        assert isinstance(k, int) and k >= 1
        return [xs[k - 1]] if len(xs) >= k else []

    def _unique(self, y: str) -> Event:
        ys = self.occ(y)
        if len(ys) != 1:
            raise ValueError(f"reference event {y!r} must occur exactly once, found {len(ys)}")
        return ys[0]

    def after(self, x: str, y: str) -> list[Event]:
        ref = self._unique(y)
        return [e for e in self.occ(x) if e.onset >= ref.offset]

    def before(self, x: str, y: str) -> list[Event]:
        ref = self._unique(y)
        return [e for e in self.occ(x) if e.offset <= ref.onset]

    def next_after(self, x: str, y: str) -> list[Event]:
        return self.after(x, y)[:1]

    def while_(self, x: str, y: str) -> list[Event]:
        if x == y:
            raise ValueError("WHILE requires distinct labels")
        ys = self.occ(y)
        return [e for e in self.occ(x) if any(e.overlap(yy) > 0 for yy in ys)]

    def not_followed(self, x: str, y: str, window: float) -> list[Event]:
        ys = self.occ(y)
        return [e for e in self.occ(x) if not any(e.offset <= yy.onset <= e.offset + window for yy in ys)]

    def absent(self, x: str) -> list[Event]:
        if self.occ(x):
            raise ValueError(f"{x!r} is present")
        return []

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {"duration": self.duration, "events": [{"label": e.label, "onset": e.onset, "offset": e.offset} for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        return cls(d["duration"], [Event(e["label"], e["onset"], e["offset"]) for e in d["events"]])
