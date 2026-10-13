"""Rank defect-risk factors by their average contribution across files. Ported from TypeScript; adapted from proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/defect-intelligence/predict.ts (MIT)."""

META = {
    "entry": "aggregate_risk_factors",
    "origin": "proffesor-for-testing/agentic-qe@214a5bc881fe4ad6324e243c9d9c2c24baae5295:src/mcp/tools/defect-intelligence/predict.ts",
    "tags": ["defect-prediction", "grouping", "sorting"],
    "domain": "software-engineering",
}


def aggregate_risk_factors(predictions):
    """Aggregate per-file risk factors into the top five factors overall.

    predictions is a list (one item per file) of lists of [factor_name, contribution]
    pairs, where contribution is an int. Every occurrence counts, even if a factor
    repeats within one file. For each factor name: count = number of occurrences and
    impact = round(sum of contributions / count, 2). Returns [name, impact, count]
    rows sorted by impact descending; ties keep the order in which the factor names
    first appeared. At most the first 5 rows are returned.
    """
    totals = {}
    for factors in predictions:
        for name, contribution in factors:
            impact, count = totals.get(name, [0, 0])
            totals[name] = [impact + contribution, count + 1]
    ranked = []
    for name, (impact, count) in totals.items():
        ranked.append([name, round(impact / count, 2), count])
    ranked.sort(key=lambda row: row[1], reverse=True)
    return ranked[:5]


def gen_inputs(rng, n):
    """Return exactly n argument lists: boundary cases first, then rng-driven cases."""
    cases = [
        [[]],
        [[[]]],
        [[[["churn", 30]]]],
        [[[["churn", 30]], [["churn", 10]]]],
        [[[["churn", 30], ["churn", 0]]]],
        [[[["a", 10], ["b", 20]], [["b", 0], ["c", 5]]]],
        [[[["a", 7], ["b", 7], ["c", 7]]]],
        [[[["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6]]]],
        [[[["a", 6], ["b", 5], ["c", 4], ["d", 3], ["e", 2], ["f", 1]]]],
        [[[["a", 1], ["b", 1], ["c", 1], ["d", 1], ["e", 1], ["f", 1]]]],
        [[[["x", 10]], [["x", 11]], [["x", 11]]]],
        [[[["size", -5], ["age", 3]]]],
    ]
    names = ["churn", "size", "age", "authors", "bugs", "deps", "lint"]
    while len(cases) < n:
        cases.append([[[[rng.choice(names), rng.randint(0, 50)] for _ in range(rng.randint(0, 3))]
                       for _ in range(rng.randint(1, 4))]])
    return cases[:n]
