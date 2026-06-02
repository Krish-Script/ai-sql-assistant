"""
collect.py -- Task 2: Collect 50-80 (NL -> SQL) pairs from Spider + WikiSQL
Saves a balanced, scenario-diverse set to data/processed/pairs.json

Run:
    python collect.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

# -- Import from your loader.py (must be in the same folder) --
sys.path.insert(0, str(Path(__file__).parent))
from loader import load_all, sample_by_scenario, SCENARIOS


# -------------------------------------------------------
# CONFIG -- edit these paths to match your folder layout
# -------------------------------------------------------

SPIDER_SPLITS  = [
    r"data/raw/spider_data/train_spider.json",
    r"data/raw/spider_data/dev.json",
]
SPIDER_TABLES  = r"data/raw/spider_data/tables.json"

WIKISQL_SPLITS = [
    r"data/raw/wikisql/train/train.parquet",
    r"data/raw/wikisql/validation/validation.parquet",
]

OUTPUT_PATH = Path("data/processed/pairs.json")

# How many pairs to pick per scenario (6 scenarios x 11 = 66 pairs total)
PAIRS_PER_SCENARIO = 11

# For reproducibility
SEED = 42


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def pick_balanced(scenario_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    From a scenario bucket, pick n rows balanced across Spider and WikiSQL.
    e.g. n=11 -> 6 from spider, 5 from wikisql (or all from one if the other
    doesn't have enough).
    """
    spider_df  = scenario_df[scenario_df["source"] == "spider"]
    wikisql_df = scenario_df[scenario_df["source"] == "wikisql"]

    half = n // 2

    spider_n  = min(half + (n % 2), len(spider_df))   # slightly more spider (richer schema)
    wikisql_n = min(n - spider_n,   len(wikisql_df))
    spider_n  = min(n - wikisql_n,  len(spider_df))    # rebalance if wikisql was short

    parts = []
    if spider_n > 0:
        parts.append(spider_df.sample(spider_n, random_state=seed))
    if wikisql_n > 0:
        parts.append(wikisql_df.sample(wikisql_n, random_state=seed))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def make_serializable(obj):
    """Recursively convert numpy/pandas types to plain Python for JSON export."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a clean list of dicts for JSON export."""
    records = []
    for _, row in df.iterrows():
        records.append({
            "nl_query"  : row["nl_query"],
            "sql_query" : row["sql_query"],
            "schema"    : make_serializable(row["schema"]),
            "scenario"  : row["scenario"],
            "difficulty": row["difficulty"],
            "source"    : row["source"],
        })
    return records


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    # 1. Load all data
    print("Loading datasets...")
    df = load_all(
        spider_splits=SPIDER_SPLITS,
        spider_tables=SPIDER_TABLES,
        wikisql_splits=WIKISQL_SPLITS,
    )
    print(f"Total loaded: {len(df):,} pairs\n")

    # 2. Sample per scenario (large pool first)
    pool = sample_by_scenario(df, n=200, seed=SEED)

    # 3. Pick balanced subset per scenario
    print(f"Collecting {PAIRS_PER_SCENARIO} pairs per scenario ({len(SCENARIOS)} scenarios)...")
    print("-" * 50)

    all_records = []
    for scenario in SCENARIOS:
        bucket = pool.get(scenario, pd.DataFrame())
        if bucket.empty:
            print(f"  {scenario:15s} -> SKIPPED (no data)")
            continue

        picked = pick_balanced(bucket, n=PAIRS_PER_SCENARIO, seed=SEED)
        records = df_to_records(picked)
        all_records.extend(records)

        # Show source breakdown
        spider_c  = sum(1 for r in records if r["source"] == "spider")
        wikisql_c = sum(1 for r in records if r["source"] == "wikisql")
        print(f"  {scenario:15s} -> {len(records):3d} pairs  "
              f"(spider: {spider_c}, wikisql: {wikisql_c})")

    print("-" * 50)
    print(f"Total collected: {len(all_records)} pairs")

    # 4. Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUTPUT_PATH.resolve()}")

    # 5. Preview 1 example per scenario
    print("\nPreview (1 per scenario):")
    print("=" * 60)
    seen = set()
    for record in all_records:
        sc = record["scenario"]
        if sc not in seen:
            seen.add(sc)
            print(f"\n[{sc.upper()}]")
            print(f"  NL  : {record['nl_query']}")
            print(f"  SQL : {record['sql_query']}")
            print(f"  Src : {record['source']}  |  Difficulty: {record['difficulty']}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()