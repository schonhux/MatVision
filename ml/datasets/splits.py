"""
ml/datasets/splits.py — leakage-safe train/val/test splitting.

This is the single most important correctness guarantee in the whole ML pipeline,
and the easiest thing to get quietly wrong. From PROJECT_GUIDE.md Section 11:

    "Never split clips from the same match across train and test — same athletes,
     uniforms, mat, camera, and lighting cause leakage. Split by athlete, match,
     venue, competition, and camera setup. The strongest test set contains athletes
     and locations absent from training."

Why this matters concretely: if two clips from the same match land on opposite sides
of the split, a model can score well by memorizing "the wrestler in the blue singlet
on this particular mat" rather than learning what a shot attempt looks like. The
resulting metric is inflated and meaningless — and worse, it looks like success.

The functions here operate on plain dicts (no DB, no torch), so they're fully
unit-testable and the guarantees are actually verified rather than assumed.

Grouping key: a sample's "group" is the set of entities it shares with other samples.
Two samples in the same group MUST end up in the same split. We group by:
  - match_id       (obviously — same footage)
  - every athlete  (same wrestler appearing in multiple matches)
  - venue          (same mat/lighting/camera position)
Groups are merged transitively: if match A and match B share an athlete, and match B
and match C share a venue, then A, B, and C all land in the same split together.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field


TRAIN, VAL, TEST = "train", "val", "test"


@dataclass
class MatchRecord:
    """The minimum metadata needed to split a match safely."""
    match_id: str
    athletes: list[str] = field(default_factory=list)  # normalized athlete identifiers
    venue: str | None = None


class _UnionFind:
    """Standard union-find. Used to merge matches into connected groups via any
    shared athlete or venue, transitively.
    """

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def normalize_athlete(name: str) -> str:
    """Athlete names are typed by hand and will be inconsistent ('J. Smith',
    'john smith', 'John  Smith'). Normalizing prevents the same wrestler from being
    treated as two people — which would silently defeat the whole point of splitting
    by athlete.
    """
    return " ".join(name.lower().replace(".", " ").replace(",", " ").split())


def build_groups(matches: list[MatchRecord]) -> dict[str, str]:
    """Assign every match to a group id. Matches sharing an athlete or venue
    (transitively) get the same group.

    Returns: {match_id: group_id}
    """
    uf = _UnionFind()

    # Seed every match as its own group.
    for m in matches:
        uf.find(f"match:{m.match_id}")

    # Link matches that share an athlete.
    by_athlete: dict[str, list[str]] = defaultdict(list)
    for m in matches:
        for athlete in m.athletes:
            by_athlete[normalize_athlete(athlete)].append(m.match_id)
    for _athlete, match_ids in by_athlete.items():
        for other in match_ids[1:]:
            uf.union(f"match:{match_ids[0]}", f"match:{other}")

    # Link matches that share a venue.
    by_venue: dict[str, list[str]] = defaultdict(list)
    for m in matches:
        if m.venue:
            by_venue[m.venue.strip().lower()].append(m.match_id)
    for _venue, match_ids in by_venue.items():
        for other in match_ids[1:]:
            uf.union(f"match:{match_ids[0]}", f"match:{other}")

    return {m.match_id: uf.find(f"match:{m.match_id}") for m in matches}


def _stable_hash(value: str, seed: int) -> float:
    """Deterministic hash → float in [0, 1). Used instead of random.shuffle so a
    given match always lands in the same split across runs and machines, which
    keeps evaluation comparable over time. (Python's hash() is salted per-process
    and would NOT be stable.)
    """
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_splits(
    matches: list[MatchRecord],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each match to train/val/test, keeping whole groups together.

    Args:
        matches: the matches to split.
        ratios: (train, val, test) target proportions — approximate, since whole
            groups move together and groups vary in size.
        seed: changes the assignment deterministically.

    Returns: {match_id: 'train' | 'val' | 'test'}
    """
    if not matches:
        return {}
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} summing to {sum(ratios)}")

    groups = build_groups(matches)
    group_ids = sorted(set(groups.values()))

    # Sort groups by a stable hash so assignment is deterministic but not correlated
    # with insertion order.
    ordered = sorted(group_ids, key=lambda g: _stable_hash(g, seed))

    # Weight by group size so the resulting split proportions track the requested
    # ratios by *match count*, not by group count (groups differ wildly in size).
    size_by_group: dict[str, int] = defaultdict(int)
    for _match_id, gid in groups.items():
        size_by_group[gid] += 1
    total = sum(size_by_group.values())

    train_target = ratios[0] * total
    val_target = ratios[1] * total

    assignment: dict[str, str] = {}
    running = 0
    for gid in ordered:
        if running < train_target:
            split = TRAIN
        elif running < train_target + val_target:
            split = VAL
        else:
            split = TEST
        for match_id, g in groups.items():
            if g == gid:
                assignment[match_id] = split
        running += size_by_group[gid]

    return assignment


def verify_no_leakage(
    matches: list[MatchRecord],
    assignment: dict[str, str],
) -> list[str]:
    """Independently re-check a split for leakage. Returns a list of human-readable
    violations; an empty list means the split is clean.

    Deliberately implemented WITHOUT reusing build_groups(), so it's a genuine
    independent check rather than a restatement of the same logic — if the grouping
    code has a bug, this should still catch it.
    """
    violations: list[str] = []

    # No athlete may appear in more than one split.
    splits_by_athlete: dict[str, set[str]] = defaultdict(set)
    for m in matches:
        split = assignment.get(m.match_id)
        if split is None:
            violations.append(f"match {m.match_id} has no split assignment")
            continue
        for athlete in m.athletes:
            splits_by_athlete[normalize_athlete(athlete)].add(split)
    for athlete, splits in splits_by_athlete.items():
        if len(splits) > 1:
            violations.append(
                f"athlete '{athlete}' appears in multiple splits: {sorted(splits)}"
            )

    # No venue may appear in more than one split.
    splits_by_venue: dict[str, set[str]] = defaultdict(set)
    for m in matches:
        split = assignment.get(m.match_id)
        if m.venue and split:
            splits_by_venue[m.venue.strip().lower()].add(split)
    for venue, splits in splits_by_venue.items():
        if len(splits) > 1:
            violations.append(f"venue '{venue}' appears in multiple splits: {sorted(splits)}")

    return violations


def split_summary(assignment: dict[str, str]) -> dict[str, int]:
    counts = {TRAIN: 0, VAL: 0, TEST: 0}
    for split in assignment.values():
        counts[split] = counts.get(split, 0) + 1
    return counts
