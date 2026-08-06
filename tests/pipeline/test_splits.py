"""
Leakage-safe split tests. If these pass, every model metric from Layer 4 onward is
measured on genuinely unseen athletes and venues. If they fail, every downstream
number is inflated and meaningless — so these are treated as critical.
"""

import pytest

from ml.datasets.splits import (
    MatchRecord,
    normalize_athlete,
    build_groups,
    assign_splits,
    verify_no_leakage,
    split_summary,
    TRAIN, VAL, TEST,
)


# --- athlete name normalization ------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("John Smith", "john smith"),
        ("john  smith", "john smith"),
        ("J. Smith", "j smith"),
        ("SMITH, JOHN", "smith john"),
        ("  Schon Huxley  ", "schon huxley"),
    ],
)
def test_normalize_athlete(raw, expected):
    assert normalize_athlete(raw) == expected


def test_name_variants_group_together():
    """The same wrestler typed inconsistently must NOT be treated as two people —
    that would silently split one athlete across train and test.
    """
    matches = [
        MatchRecord("m1", athletes=["John Smith"]),
        MatchRecord("m2", athletes=["john  smith"]),
        MatchRecord("m3", athletes=["J. Smith"]),  # different person by our rules
    ]
    groups = build_groups(matches)
    assert groups["m1"] == groups["m2"], "name variants of the same athlete must group together"


# --- grouping -----------------------------------------------------------------

def test_matches_sharing_an_athlete_are_grouped():
    matches = [
        MatchRecord("m1", athletes=["schon"]),
        MatchRecord("m2", athletes=["schon", "opponent_a"]),
        MatchRecord("m3", athletes=["unrelated"]),
    ]
    groups = build_groups(matches)
    assert groups["m1"] == groups["m2"]
    assert groups["m3"] != groups["m1"]


def test_matches_sharing_a_venue_are_grouped():
    matches = [
        MatchRecord("m1", athletes=["a"], venue="Hilton Coliseum"),
        MatchRecord("m2", athletes=["b"], venue="hilton coliseum"),  # case-insensitive
        MatchRecord("m3", athletes=["c"], venue="Other Gym"),
    ]
    groups = build_groups(matches)
    assert groups["m1"] == groups["m2"]
    assert groups["m3"] != groups["m1"]


def test_grouping_is_transitive():
    """A-B share an athlete, B-C share a venue => A, B, C must all be one group.
    This is the case a naive implementation gets wrong.
    """
    matches = [
        MatchRecord("mA", athletes=["schon"], venue="gym_1"),
        MatchRecord("mB", athletes=["schon"], venue="gym_2"),
        MatchRecord("mC", athletes=["stranger"], venue="gym_2"),
    ]
    groups = build_groups(matches)
    assert groups["mA"] == groups["mB"] == groups["mC"]


def test_unrelated_matches_stay_separate():
    matches = [
        MatchRecord("m1", athletes=["a"], venue="v1"),
        MatchRecord("m2", athletes=["b"], venue="v2"),
    ]
    groups = build_groups(matches)
    assert groups["m1"] != groups["m2"]


def test_match_with_no_metadata_is_its_own_group():
    matches = [MatchRecord("m1"), MatchRecord("m2")]
    groups = build_groups(matches)
    assert groups["m1"] != groups["m2"]


# --- split assignment ----------------------------------------------------------

def test_empty_input():
    assert assign_splits([]) == {}


def test_ratios_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        assign_splits([MatchRecord("m1")], ratios=(0.5, 0.2, 0.2))


def test_every_match_gets_a_split():
    matches = [MatchRecord(f"m{i}", athletes=[f"a{i}"]) for i in range(20)]
    assignment = assign_splits(matches)
    assert set(assignment.keys()) == {m.match_id for m in matches}
    assert all(v in (TRAIN, VAL, TEST) for v in assignment.values())


def test_assignment_is_deterministic():
    matches = [MatchRecord(f"m{i}", athletes=[f"a{i}"]) for i in range(30)]
    first = assign_splits(matches, seed=7)
    second = assign_splits(matches, seed=7)
    assert first == second, "same seed must produce identical splits across runs"


def test_different_seeds_produce_different_splits():
    matches = [MatchRecord(f"m{i}", athletes=[f"a{i}"]) for i in range(40)]
    assert assign_splits(matches, seed=1) != assign_splits(matches, seed=2)


def test_grouped_matches_land_in_same_split():
    """The core guarantee. Matches sharing an athlete must never be separated."""
    matches = []
    for group in range(10):
        for m in range(3):
            matches.append(
                MatchRecord(f"g{group}_m{m}", athletes=[f"wrestler_{group}"])
            )
    assignment = assign_splits(matches)
    for group in range(10):
        splits = {assignment[f"g{group}_m{m}"] for m in range(3)}
        assert len(splits) == 1, f"group {group} was split across {splits}"


def test_split_proportions_are_roughly_respected():
    matches = [MatchRecord(f"m{i}", athletes=[f"a{i}"]) for i in range(100)]
    assignment = assign_splits(matches, ratios=(0.7, 0.15, 0.15))
    counts = split_summary(assignment)
    assert counts[TRAIN] + counts[VAL] + counts[TEST] == 100
    # Generous bounds: whole groups move together, so exact ratios aren't possible.
    assert 55 <= counts[TRAIN] <= 85, counts
    assert counts[TEST] >= 5, counts


# --- independent leakage verification -------------------------------------------

def test_verify_no_leakage_passes_on_a_correct_split():
    matches = [
        MatchRecord(f"m{i}", athletes=[f"wrestler_{i // 3}"], venue=f"gym_{i // 5}")
        for i in range(30)
    ]
    assignment = assign_splits(matches)
    assert verify_no_leakage(matches, assignment) == []


def test_verify_no_leakage_catches_athlete_across_splits():
    """A deliberately broken split must be caught — this proves the verifier
    actually works rather than trivially returning [].
    """
    matches = [
        MatchRecord("m1", athletes=["schon"]),
        MatchRecord("m2", athletes=["schon"]),
    ]
    bad = {"m1": TRAIN, "m2": TEST}
    violations = verify_no_leakage(matches, bad)
    assert any("schon" in v for v in violations)


def test_verify_no_leakage_catches_venue_across_splits():
    matches = [
        MatchRecord("m1", athletes=["a"], venue="Hilton"),
        MatchRecord("m2", athletes=["b"], venue="Hilton"),
    ]
    bad = {"m1": TRAIN, "m2": VAL}
    violations = verify_no_leakage(matches, bad)
    assert any("hilton" in v.lower() for v in violations)


def test_verify_no_leakage_catches_missing_assignment():
    matches = [MatchRecord("m1", athletes=["a"]), MatchRecord("m2", athletes=["b"])]
    violations = verify_no_leakage(matches, {"m1": TRAIN})
    assert any("m2" in v for v in violations)


def test_realistic_scenario_end_to_end():
    """A plausible early dataset: the same wrestler (Schon) in many matches across a
    few venues, plus various opponents. The whole thing should collapse into few
    groups, and must still verify clean.
    """
    matches = [
        MatchRecord("m1", athletes=["Schon Huxley", "Opponent A"], venue="Hilton Coliseum"),
        MatchRecord("m2", athletes=["Schon Huxley", "Opponent B"], venue="Hilton Coliseum"),
        MatchRecord("m3", athletes=["schon huxley", "Opponent C"], venue="Away Gym"),
        MatchRecord("m4", athletes=["Teammate D", "Opponent E"], venue="Practice Room"),
        MatchRecord("m5", athletes=["Teammate F", "Opponent G"], venue="Practice Room"),
        MatchRecord("m6", athletes=["Stranger H", "Stranger I"], venue="Neutral Site"),
    ]
    assignment = assign_splits(matches)
    assert verify_no_leakage(matches, assignment) == []

    # All Schon matches share an athlete -> one group, one split.
    assert len({assignment["m1"], assignment["m2"], assignment["m3"]}) == 1
    # Practice-room matches share a venue -> one split.
    assert assignment["m4"] == assignment["m5"]
