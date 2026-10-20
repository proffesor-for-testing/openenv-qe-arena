"""Math family checks beyond the generic verifier tests: references agree with independent brute force, integers are
strict, and level 3 serves its special outcomes (inconsistent systems, claims that hold) on a fair share of seeds."""

from __future__ import annotations

import json
from itertools import permutations
from math import gcd

from qe_arena import skills
from qe_arena.skills import formal

SEEDS = range(120)


def _tasks(level: int, name: str):
    return [skills.make_skill_task(f"qe_skill-math-d{level}", s, pool=(name,)) for s in SEEDS]


def _ok(arr: tuple[str, ...], c: dict) -> bool:
    if c["type"] == "not_at":
        return arr[c["position"] - 1] != c["item"]
    x, y = c["items"]
    if c["type"] == "not_adjacent":
        return all(arr[i:i + 2] not in ((x, y), (y, x)) for i in range(len(arr) - 1))
    return arr.index(x) < arr.index(y)


def test_counts_match_a_direct_enumeration_of_distinct_rows():
    for level in (1, 2, 3):
        for task in _tasks(level, "counting"):
            problem = json.loads(task.files["problem.json"])
            rows = {p for p in permutations(problem["items"]) if all(_ok(p, c) for c in problem["constraints"])}
            assert task.private["reference"] == {"count": len(rows)}
            assert len(problem["constraints"]) == level
            assert level == 1 or len(set(problem["items"])) < len(problem["items"])


def test_congruence_answers_match_a_scan_of_one_full_period():
    statuses = set()
    for level in (1, 2, 3):
        for task in _tasks(level, "congruences"):
            pairs = [(c["remainder"], c["modulus"]) for c in json.loads(task.files["system.json"])["congruences"]]
            period = 1
            for _, m in pairs:
                period = period * m // gcd(period, m)
            hits = [x for x in range(period) if all(x % m == r for r, m in pairs)]
            ref = task.private["reference"]
            statuses.add((level, ref["status"]))
            assert ref == ({"status": "none"} if not hits else {"status": "solved", "x": hits[0], "modulus": period})
            assert len(hits) <= 1
    assert (3, "none") in statuses and (3, "solved") in statuses and (2, "none") not in statuses


def test_falsify_answers_match_a_naive_scan_and_level_three_mixes_outcomes():
    results = []
    for level in (1, 2, 3):
        for task in _tasks(level, "falsify"):
            doc = json.loads(task.files["claim.json"])
            spans = [range(lo, hi + 1) for lo, hi in doc["ranges"].values()]
            points = [(n,) for n in spans[0]] if len(spans) == 1 else [(a, b) for a in spans[0] for b in spans[1]]
            bad = [p for p in points if not formal._holds(doc["claim"], dict(zip(doc["variables"], p)))]
            expected = ({"result": "counterexample", **dict(zip(doc["variables"], min(bad)))} if bad
                        else {"result": "holds"})
            assert task.private["reference"] == expected
            results.append((level, expected["result"]))
    share = results.count((3, "holds")) / len(SEEDS)
    assert 0.15 < share < 0.5 and (1, "holds") not in results and (2, "holds") not in results


def test_integers_must_be_json_integers():
    for name, key in (("counting", "count"), ("congruences", "x"), ("falsify", None)):
        task = next(t for t in _tasks(2, name))
        out = json.loads(skills.solve(task))
        key = key or next(k for k in out if k != "result")
        for wrong in (float(out[key]), str(out[key]), True if out[key] == 1 else out[key] + 1):
            assert skills.grade(task, json.dumps(dict(out, **{key: wrong}))) == 0.0, (name, wrong)
        assert skills.grade(task, json.dumps(out)) == 1.0
