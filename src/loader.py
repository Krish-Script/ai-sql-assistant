"""
loader.py — Dataset loader for Spider (JSON) and WikiSQL (Parquet)
Outputs a unified DataFrame with columns:
    nl_query    : natural language question
    sql_query   : SQL string
    schema      : table/column info (dict or string)
    difficulty  : 'easy' | 'medium' | 'hard' (Spider) or 'unknown' (WikiSQL)
    source      : 'spider' | 'wikisql'
    scenario    : 'filters' | 'aggregations' | 'joins' | 'subqueries' |
                  'sort_limit' | 'simple_select'
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ----------------------------------------------
# Scenario tagger (shared by both loaders)
# ----------------------------------------------

def tag_scenario(sql: str) -> str:
    """Classify a SQL string into one of 5 scenario buckets."""
    q = sql.upper()
    if re.search(r'\bJOIN\b', q):
        return "joins"
    if re.search(r'\b(SELECT\s.+\bIN\s*\(SELECT|EXISTS\s*\(SELECT)', q):
        return "subqueries"
    if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', q):
        return "aggregations"
    if re.search(r'\bORDER\s+BY\b', q) or re.search(r'\bLIMIT\b', q):
        return "sort_limit"
    if re.search(r'\bWHERE\b', q):
        return "filters"
    return "simple_select"


# ----------------------------------------------
# Spider loader
# ----------------------------------------------

def _build_spider_schema(db_id: str, tables_json: list[dict]) -> dict:
    """Look up schema info for a given db_id from tables.json entries."""
    for entry in tables_json:
        if entry["db_id"] == db_id:
            return {
                "db_id": db_id,
                "table_names": entry.get("table_names_original", []),
                "column_names": entry.get("column_names_original", []),
            }
    return {"db_id": db_id}


def load_spider(
    split_path: str | Path,
    tables_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load one Spider split file (train_spider.json / dev.json / test.json).

    Args:
        split_path  : path to e.g. data/raw/spider/train_spider.json
        tables_path : path to tables.json (optional — adds schema info)

    Returns:
        Unified DataFrame.
    """
    split_path = Path(split_path)
    with open(split_path, encoding="utf-8") as f:
        data = json.load(f)

    tables_json: list[dict] = []
    if tables_path and Path(tables_path).exists():
        with open(tables_path, encoding="utf-8") as f:
            tables_json = json.load(f)

    rows = []
    for item in data:
        db_id = item.get("db_id", "")
        sql   = item.get("query", "")
        nl    = item.get("question", "")

        schema = (
            _build_spider_schema(db_id, tables_json)
            if tables_json
            else {"db_id": db_id}
        )

        rows.append({
            "nl_query"  : nl,
            "sql_query" : sql,
            "schema"    : schema,
            "difficulty": item.get("query_toks_no_value", {})  # placeholder
                            if False else item.get("difficulty", "unknown"),
            "source"    : "spider",
            "scenario"  : tag_scenario(sql),
        })

    df = pd.DataFrame(rows)
    assert {"nl_query", "sql_query"}.issubset(df.columns), "Spider: missing columns"
    return df


# ----------------------------------------------
# WikiSQL loader
# ----------------------------------------------

def load_wikisql(parquet_path: str | Path) -> pd.DataFrame:
    """
    Load a WikiSQL parquet split.

    Expected columns in parquet: phase, question, table, sql
    The `sql` column is a dict with key 'human_readable'.

    Args:
        parquet_path : path to e.g. data/raw/wikisql/train.parquet

    Returns:
        Unified DataFrame.
    """
    parquet_path = Path(parquet_path)
    raw = pd.read_parquet(parquet_path)

    required = {"question", "sql"}
    assert required.issubset(raw.columns), (
        f"WikiSQL parquet missing columns. Found: {raw.columns.tolist()}"
    )

    rows = []
    for _, row in raw.iterrows():
        sql_dict = row["sql"]

        # Support both dict-style and plain-string sql column
        if isinstance(sql_dict, dict):
            sql = sql_dict.get("human_readable", "")
        else:
            sql = str(sql_dict)

        # table column may be a dict or string
        table_info = row.get("table", {})
        schema = (
            {"table_id": table_info.get("id", ""), "header": table_info.get("header", [])}
            if isinstance(table_info, dict)
            else {"table_id": str(table_info)}
        )

        rows.append({
            "nl_query"  : row["question"],
            "sql_query" : sql,
            "schema"    : schema,
            "difficulty": "unknown",   # WikiSQL has no difficulty labels
            "source"    : "wikisql",
            "scenario"  : tag_scenario(sql),
        })

    df = pd.DataFrame(rows)
    assert {"nl_query", "sql_query"}.issubset(df.columns), "WikiSQL: missing columns"
    return df


# ----------------------------------------------
# Combined loader
# ----------------------------------------------

def load_all(
    spider_splits: list[str | Path] | None = None,
    spider_tables: str | Path | None = None,
    wikisql_splits: list[str | Path] | None = None,
) -> pd.DataFrame:
    """
    Load and combine Spider + WikiSQL splits into one DataFrame.

    Args:
        spider_splits  : list of Spider split JSON paths
        spider_tables  : path to Spider's tables.json
        wikisql_splits : list of WikiSQL parquet paths

    Returns:
        Combined DataFrame with unified schema.
    """
    frames: list[pd.DataFrame] = []

    for path in (spider_splits or []):
        print(f"[Spider]   Loading {path} ...")
        frames.append(load_spider(path, tables_path=spider_tables))

    for path in (wikisql_splits or []):
        print(f"[WikiSQL]  Loading {path} ...")
        frames.append(load_wikisql(path))

    if not frames:
        raise ValueError("No data loaded — provide at least one split path.")

    df = pd.concat(frames, ignore_index=True)
    return df


# ----------------------------------------------
# Sampler — Task 1: sample per scenario bucket
# ----------------------------------------------

SCENARIOS = ["filters", "aggregations", "joins", "subqueries", "sort_limit", "simple_select"]

def sample_by_scenario(
    df: pd.DataFrame,
    n: int = 100,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Sample up to `n` examples per scenario bucket.

    Returns:
        Dict mapping scenario name -> sampled DataFrame.
    """
    random.seed(seed)
    buckets: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        subset = df[df["scenario"] == scenario]
        k = min(n, len(subset))
        buckets[scenario] = subset.sample(k, random_state=seed) if k > 0 else subset
    return buckets


# ----------------------------------------------
# Quick smoke-test
# ----------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load Spider + WikiSQL datasets")
    parser.add_argument("--spider",        nargs="*", default=[], help="Spider JSON split paths")
    parser.add_argument("--spider-tables", default=None,          help="Path to Spider tables.json")
    parser.add_argument("--wikisql",       nargs="*", default=[], help="WikiSQL parquet split paths")
    parser.add_argument("--sample",        type=int,  default=100, help="Examples per scenario")
    args = parser.parse_args()

    # -- fallback defaults (edit these to match your folder layout) --
    spider_splits = args.spider or [
        "data/raw/spider_data/train_spider.json",
        "data/raw/spider_data/dev.json",
    ]
    spider_tables = args.spider_tables or "data/raw/spider_data/tables.json"
    wikisql_splits = args.wikisql or [
        "data/raw/wikisql/train/train.parquet",
        "data/raw/wikisql/validation/validation.parquet",
    ]

    df = load_all(
        spider_splits=spider_splits,
        spider_tables=spider_tables,
        wikisql_splits=wikisql_splits,
    )

    print(f"\n{'-'*50}")
    print(f"Total pairs loaded : {len(df):,}")
    print(f"Sources            : {df['source'].value_counts().to_dict()}")
    print(f"\nScenario distribution:")
    print(df["scenario"].value_counts().to_string())
    print(f"{'-'*50}\n")

    buckets = sample_by_scenario(df, n=args.sample)

    print("Sampled buckets:")
    for scenario, sdf in buckets.items():
        print(f"  {scenario:15s} -> {len(sdf):4d} examples")

    print("\nSample row (filters):")
    if len(buckets.get("filters", pd.DataFrame())) > 0:
        row = buckets["filters"].iloc[0]
        print(f"  NL  : {row['nl_query']}")
        print(f"  SQL : {row['sql_query']}")
        print(f"  Src : {row['source']}  |  Difficulty: {row['difficulty']}")