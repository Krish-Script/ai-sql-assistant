"""
clean.py -- Task 3: Clean and normalize pairs.json into a structured format
Reads  : data/processed/pairs.json
Writes : data/processed/pairs_clean.json
         data/processed/pairs_clean.csv

Cleaning steps:
    1. Drop duplicate SQL queries
    2. Normalize SQL keywords to UPPERCASE
    3. Normalize whitespace in NL and SQL
    4. Flatten schema into a consistent structure
    5. Derive difficulty from SQL complexity
    6. Validate every row passes minimum quality bar
    7. Save as both JSON and CSV
"""

import json
import re
import csv
import sys
from pathlib import Path

# OutputParser for ground truth normalization (Week 4 fix)
sys.path.insert(0, str(Path(__file__).parent))
try:
    from output_parser import OutputParser as _OutputParser
    _gt_parser = _OutputParser()
    _HAS_PARSER = True
except ImportError:
    _HAS_PARSER = False


INPUT_PATH       = Path("data/processed/pairs.json")
OUTPUT_JSON_PATH = Path("data/processed/pairs_clean.json")
OUTPUT_CSV_PATH  = Path("data/processed/pairs_clean.csv")


# -------------------------------------------------------
# Step 1 -- SQL keyword normalizer
# -------------------------------------------------------

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "ON", "AS", "DISTINCT", "ORDER", "BY", "GROUP", "HAVING",
    "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT", "CASE",
    "WHEN", "THEN", "ELSE", "END", "NULL", "IS", "LIKE", "BETWEEN",
    "COUNT", "SUM", "AVG", "MAX", "MIN", "ASC", "DESC", "INSERT",
    "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "TABLE", "INTO",
    "VALUES", "SET", "PRIMARY", "KEY", "FOREIGN", "REFERENCES",
]

def normalize_sql(sql: str) -> str:
    """
    Uppercase SQL keywords while preserving string literals and identifiers.
    Also collapses extra whitespace.
    """
    # Collapse all whitespace first
    sql = re.sub(r'\s+', ' ', sql.strip())

    # Uppercase known keywords (whole-word match, case-insensitive)
    for kw in SQL_KEYWORDS:
        sql = re.sub(rf'\b{kw}\b', kw, sql, flags=re.IGNORECASE)

    return sql


# -------------------------------------------------------
# Step 2 -- Schema normalizer
# -------------------------------------------------------

def normalize_schema(schema: dict, source: str) -> dict:
    """
    Flatten source-specific schema dicts into a unified shape:
    {
        "tables"  : ["table1", "table2"],
        "columns" : ["col1", "col2", ...],
        "raw_id"  : "original db_id or table_id"
    }
    """
    if source == "spider":
        raw_id  = schema.get("db_id", "")
        tables  = schema.get("table_names", [])
        # column_names is [[table_idx, col_name], ...] -- extract just names
        col_raw = schema.get("column_names", [])
        columns = [c[1] for c in col_raw if isinstance(c, list) and len(c) == 2 and c[1] != "*"]

    elif source == "wikisql":
        raw_id  = schema.get("table_id", "")
        tables  = ["table"]                      # WikiSQL is always single-table
        columns = schema.get("header", [])

    else:
        raw_id  = ""
        tables  = []
        columns = []

    return {
        "raw_id" : raw_id,
        "tables" : tables,
        "columns": columns,
    }


# -------------------------------------------------------
# Step 3 -- Difficulty deriver
# -------------------------------------------------------

def derive_difficulty(sql: str) -> str:
    """
    Derive difficulty from SQL structure since labels are missing.
    easy   : simple SELECT, single table, basic WHERE
    medium : aggregations, GROUP BY, ORDER BY, LIMIT
    hard   : JOINs, subqueries, UNION/INTERSECT/EXCEPT, HAVING
    """
    q = sql.upper()
    hard_patterns = [
        r'\bJOIN\b', r'\bIN\s*\(SELECT', r'\bEXISTS\s*\(',
        r'\bUNION\b', r'\bINTERSECT\b', r'\bEXCEPT\b', r'\bHAVING\b',
    ]
    medium_patterns = [
        r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(',
        r'\bGROUP\s+BY\b', r'\bORDER\s+BY\b', r'\bLIMIT\b',
    ]
    if any(re.search(p, q) for p in hard_patterns):
        return "hard"
    if any(re.search(p, q) for p in medium_patterns):
        return "medium"
    return "easy"


# -------------------------------------------------------
# Step 4 -- Row validator
# -------------------------------------------------------

def validate_row(row: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    A row is invalid if it fails any minimum quality check.
    """
    if not row.get("nl_query", "").strip():
        return False, "empty nl_query"
    if not row.get("sql_query", "").strip():
        return False, "empty sql_query"
    if len(row["nl_query"].split()) < 3:
        return False, "nl_query too short (< 3 words)"
    if "SELECT" not in row["sql_query"].upper():
        return False, "sql_query missing SELECT"
    if len(row["sql_query"]) < 10:
        return False, "sql_query suspiciously short"
    return True, "ok"


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    # Load
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} pairs from {INPUT_PATH}")

    cleaned   = []
    dropped   = []
    seen_sqls = set()

    for i, row in enumerate(data):
        original_sql = row["sql_query"]

        # -- Step 1: Deduplicate on normalized SQL --
        dedup_key = re.sub(r'\s+', ' ', original_sql.strip().upper())
        if dedup_key in seen_sqls:
            dropped.append((i, row["nl_query"], "duplicate SQL"))
            continue
        seen_sqls.add(dedup_key)

        # -- Step 2: Normalize SQL --
        clean_sql = normalize_sql(original_sql)

        # -- Step 2b: Run expected SQL through OutputParser (Week 4 fix) --
        # This normalizes quote style, keyword casing, whitespace so that
        # ground truth SQL matches the same format as generated SQL.
        # Without this, style differences cause false PARTIAL scores.
        if _HAS_PARSER:
            parsed = _gt_parser.parse(clean_sql)
            if parsed.success:
                clean_sql = parsed.sql

        # -- Step 3: Normalize NL (strip + collapse whitespace) --
        clean_nl = re.sub(r'\s+', ' ', row["nl_query"].strip())

        # -- Step 4: Normalize schema --
        clean_schema = normalize_schema(row.get("schema", {}), row["source"])

        # -- Step 5: Derive difficulty --
        difficulty = derive_difficulty(clean_sql)

        clean_row = {
            "nl_query"  : clean_nl,
            "sql_query" : clean_sql,
            "schema"    : clean_schema,
            "scenario"  : row["scenario"],
            "difficulty": difficulty,
            "source"    : row["source"],
        }

        # -- Step 6: Validate --
        is_valid, reason = validate_row(clean_row)
        if not is_valid:
            dropped.append((i, row["nl_query"], reason))
            continue

        cleaned.append(clean_row)

    # ── Report ──
    print()
    print("-" * 50)
    print(f"Clean pairs : {len(cleaned)}")
    print(f"Dropped     : {len(dropped)}")
    if dropped:
        print("  Dropped rows:")
        for idx, nl, reason in dropped:
            print(f"    row {idx:02d} | {reason:25s} | {nl[:60]}")
    print()

    # ── Difficulty breakdown ──
    diff_counts = {}
    for r in cleaned:
        diff_counts[r["difficulty"]] = diff_counts.get(r["difficulty"], 0) + 1
    print("Difficulty breakdown:")
    for d, c in sorted(diff_counts.items()):
        print(f"  {d:8s} : {c}")
    print()

    # ── Scenario breakdown ──
    sc_counts = {}
    for r in cleaned:
        sc_counts[r["scenario"]] = sc_counts.get(r["scenario"], 0) + 1
    print("Scenario breakdown:")
    for s, c in sorted(sc_counts.items()):
        print(f"  {s:15s} : {c}")
    print("-" * 50)

    # ── Save JSON ──
    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)
    print(f"\nSaved JSON -> {OUTPUT_JSON_PATH.resolve()}")

    # ── Save CSV ──
    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "nl_query", "sql_query", "scenario", "difficulty", "source",
            "schema_tables", "schema_columns"
        ])
        writer.writeheader()
        for r in cleaned:
            writer.writerow({
                "nl_query"      : r["nl_query"],
                "sql_query"     : r["sql_query"],
                "scenario"      : r["scenario"],
                "difficulty"    : r["difficulty"],
                "source"        : r["source"],
                "schema_tables" : "|".join(r["schema"]["tables"]),
                "schema_columns": "|".join(r["schema"]["columns"]),
            })
    print(f"Saved CSV  -> {OUTPUT_CSV_PATH.resolve()}")

    # ── Preview 1 cleaned row per scenario ──
    print("\nPreview (1 per scenario after cleaning):")
    print("=" * 60)
    seen = set()
    for r in cleaned:
        sc = r["scenario"]
        if sc not in seen:
            seen.add(sc)
            print(f"\n[{sc.upper()}]  difficulty={r['difficulty']}  source={r['source']}")
            print(f"  NL  : {r['nl_query']}")
            print(f"  SQL : {r['sql_query']}")
            print(f"  Tbls: {r['schema']['tables']}")
            print(f"  Cols: {r['schema']['columns'][:5]} ...")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()