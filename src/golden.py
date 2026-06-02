"""
golden.py -- Task 5: Manually label 10 pairs as the golden evaluation set
Reads  : data/processed/pairs_clean.json
Writes : data/processed/golden.json

The golden set contains:
    - 9 correct, high-quality NL->SQL pairs (hand-picked for clarity + diversity)
    - 1 intentionally wrong pair (to verify the evaluator catches bad output)

Each pair has an extra field:
    is_correct : True for real pairs, False for the intentional wrong one
    notes      : Human annotation explaining the pair or the error type
    gold_id    : Stable ID for referencing in evaluation reports
"""

import json
from pathlib import Path

INPUT_PATH  = Path("data/processed/pairs_clean.json")
OUTPUT_PATH = Path("data/golden/golden.json")

# Row indices chosen from pairs_clean.json
# Criteria: clarity, scenario diversity, difficulty spread, real-world relevance
GOLDEN_INDICES = [0, 3, 13, 14, 22, 26, 33, 37, 42, 56]

# Human annotations for each picked row (index -> note)
NOTES = {
    0 : "Clean filter with != operator. Simple, unambiguous NL. Good easy baseline.",
    3 : "Hard filter requiring WHERE with multiple conditions across industries.",
    13: "Two aggregate functions (MAX + AVG) in one SELECT. Good medium complexity.",
    14: "GROUP BY with ORDER BY and result ordering. Real-world reporting pattern.",
    22: "3-table JOIN with ORDER BY + LIMIT 1. Classic find-the-min/max pattern.",
    26: "JOIN across customer/account tables with SUM aggregation. Very realistic.",
    33: "NOT IN subquery checking absence. Classic hard pattern for LLMs to get right.",
    37: "COUNT with NOT IN subquery. Tests whether LLM can handle negation correctly.",
    42: "ORDER BY DESC + LIMIT 1 to find top record. Common dashboard query.",
    56: "Simple SELECT DISTINCT. Clean easy baseline with no conditions.",
}

# The intentionally wrong pair -- used to verify the evaluator catches errors
WRONG_PAIR = {
    "gold_id"   : "gold_wrong_01",
    "nl_query"  : "How many airports do we have?",
    # Correct SQL would be: SELECT COUNT(*) FROM AIRPORTS
    # Error type: missing aggregate -- returns all rows instead of a count
    "sql_query" : "SELECT * FROM AIRPORTS",
    "schema"    : {
        "raw_id" : "flight_2",
        "tables" : ["airlines", "airports", "flights"],
        "columns": ["uid", "Airline", "Abbreviation", "Country", "City",
                    "AirportCode", "AirportName", "State", "id",
                    "SourceAirport", "DestAirport"],
    },
    "scenario"  : "aggregations",
    "difficulty": "medium",
    "source"    : "spider",
    "is_correct": False,
    "error_type": "missing_aggregate",
    "notes"     : (
        "INTENTIONALLY WRONG. NL asks for a COUNT but SQL returns all rows. "
        "Simulates a common LLM mistake: understanding the table but missing "
        "the aggregation. Evaluator must flag this as incorrect."
    ),
}


def main():
    # Load cleaned pairs
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} pairs from {INPUT_PATH}\n")

    # Pick golden rows
    golden = []
    for gid, idx in enumerate(GOLDEN_INDICES, start=1):
        row = data[idx].copy()
        row["gold_id"]    = f"gold_{gid:02d}"
        row["is_correct"] = True
        row["error_type"] = None
        row["notes"]      = NOTES.get(idx, "")
        golden.append(row)
        print(f"[gold_{gid:02d}] row={idx:02d}  [{row['scenario']:15s}]"
              f"  [{row['difficulty']:6s}]  {row['nl_query'][:60]}")

    # Append the intentionally wrong pair
    golden.append(WRONG_PAIR)
    print(f"\n[gold_wrong_01]  INTENTIONALLY WRONG  [{WRONG_PAIR['scenario']:15s}]"
          f"  [{WRONG_PAIR['difficulty']:6s}]  {WRONG_PAIR['nl_query']}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  Golden set summary")
    print(f"{'='*55}")
    print(f"  Total pairs     : {len(golden)}")
    print(f"  Correct pairs   : {sum(1 for g in golden if g['is_correct'])}")
    print(f"  Wrong pairs     : {sum(1 for g in golden if not g['is_correct'])}")

    print(f"\n  Scenario spread:")
    sc_counts = {}
    for g in golden:
        sc_counts[g["scenario"]] = sc_counts.get(g["scenario"], 0) + 1
    for s, c in sorted(sc_counts.items()):
        print(f"    {s:15s} : {c}")

    print(f"\n  Difficulty spread:")
    diff_counts = {}
    for g in golden:
        diff_counts[g["difficulty"]] = diff_counts.get(g["difficulty"], 0) + 1
    for d, c in sorted(diff_counts.items()):
        print(f"    {d:8s} : {c}")

    print(f"{'='*55}")
    print(f"\nSaved -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()