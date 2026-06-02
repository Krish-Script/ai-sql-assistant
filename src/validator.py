"""
validator.py -- Task 4: Data loader + validator for pairs_clean.json
Reads  : data/processed/pairs_clean.json
Outputs: validation report printed to console
         data/processed/validation_report.json

Validation checks:
    1.  Schema fields present       -- all 6 required keys exist
    2.  No empty NL query           -- nl_query is non-empty string
    3.  No empty SQL query          -- sql_query is non-empty string
    4.  NL minimum length           -- at least 3 words
    5.  SQL starts with SELECT      -- basic SQL sanity
    6.  SQL keyword casing          -- keywords must be UPPERCASE
    7.  Scenario is valid           -- must be one of the 6 known scenarios
    8.  Difficulty is valid         -- must be easy / medium / hard
    9.  Source is valid             -- must be spider or wikisql
    10. Schema has tables + columns -- schema dict must have non-empty lists
    11. No duplicate NL queries     -- nl_query must be unique across dataset
    12. No duplicate SQL queries    -- sql_query must be unique across dataset
    13. Scenario-SQL consistency    -- SQL structure matches tagged scenario
"""

import json
import re
from pathlib import Path
from collections import defaultdict


INPUT_PATH  = Path("data/processed/pairs_clean.json")
REPORT_PATH = Path("data/processed/validation_report.json")

VALID_SCENARIOS  = {"filters", "aggregations", "joins", "subqueries", "sort_limit", "simple_select"}
VALID_SOURCES    = {"spider", "wikisql"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}
REQUIRED_KEYS    = {"nl_query", "sql_query", "schema", "scenario", "difficulty", "source"}

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "ON", "AS", "DISTINCT", "ORDER", "BY", "GROUP", "HAVING",
    "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT",
    "COUNT", "SUM", "AVG", "MAX", "MIN", "ASC", "DESC",
]


# -------------------------------------------------------
# Individual check functions
# (each returns None if ok, or a string describing the issue)
# -------------------------------------------------------

def check_required_keys(row: dict) -> str | None:
    missing = REQUIRED_KEYS - set(row.keys())
    if missing:
        return f"Missing keys: {sorted(missing)}"
    return None


def check_nl_empty(row: dict) -> str | None:
    if not row.get("nl_query", "").strip():
        return "nl_query is empty"
    return None


def check_sql_empty(row: dict) -> str | None:
    if not row.get("sql_query", "").strip():
        return "sql_query is empty"
    return None


def check_nl_length(row: dict) -> str | None:
    words = row.get("nl_query", "").split()
    if len(words) < 3:
        return f"nl_query too short ({len(words)} words)"
    return None


def check_sql_has_select(row: dict) -> str | None:
    if "SELECT" not in row.get("sql_query", "").upper():
        return "sql_query missing SELECT"
    return None


def check_sql_keyword_casing(row: dict) -> str | None:
    sql = row.get("sql_query", "")
    bad = []
    for kw in SQL_KEYWORDS:
        # Find the keyword in lowercase (meaning it wasn't normalized)
        if re.search(rf'\b{kw.lower()}\b', sql):
            bad.append(kw.lower())
    if bad:
        return f"Lowercase SQL keywords found: {bad[:3]}"
    return None


def check_scenario_valid(row: dict) -> str | None:
    sc = row.get("scenario", "")
    if sc not in VALID_SCENARIOS:
        return f"Invalid scenario: '{sc}'"
    return None


def check_difficulty_valid(row: dict) -> str | None:
    d = row.get("difficulty", "")
    if d not in VALID_DIFFICULTY:
        return f"Invalid difficulty: '{d}'"
    return None


def check_source_valid(row: dict) -> str | None:
    s = row.get("source", "")
    if s not in VALID_SOURCES:
        return f"Invalid source: '{s}'"
    return None


def check_schema_structure(row: dict) -> str | None:
    schema = row.get("schema", {})
    if not isinstance(schema, dict):
        return "schema is not a dict"
    if not schema.get("tables"):
        return "schema.tables is empty"
    if not schema.get("columns"):
        return "schema.columns is empty"
    return None


def check_scenario_sql_consistency(row: dict) -> str | None:
    """Check that the SQL structure actually matches the tagged scenario."""
    sql = row.get("sql_query", "").upper()
    sc  = row.get("scenario", "")

    if sc == "joins" and not re.search(r'\bJOIN\b', sql):
        return "Scenario=joins but no JOIN in SQL"
    if sc == "subqueries" and not re.search(r'\bIN\s*\(SELECT|\bEXISTS\s*\(', sql):
        return "Scenario=subqueries but no subquery pattern in SQL"
    if sc == "aggregations" and not re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql):
        return "Scenario=aggregations but no aggregate function in SQL"
    if sc == "sort_limit" and not re.search(r'\bORDER\s+BY\b|\bLIMIT\b', sql):
        return "Scenario=sort_limit but no ORDER BY or LIMIT in SQL"
    if sc == "filters" and not re.search(r'\bWHERE\b', sql):
        return "Scenario=filters but no WHERE in SQL"
    return None


# All per-row checks in order
ROW_CHECKS = [
    check_required_keys,
    check_nl_empty,
    check_sql_empty,
    check_nl_length,
    check_sql_has_select,
    check_sql_keyword_casing,
    check_scenario_valid,
    check_difficulty_valid,
    check_source_valid,
    check_schema_structure,
    check_scenario_sql_consistency,
]


# -------------------------------------------------------
# Dataset-level checks (need all rows at once)
# -------------------------------------------------------

def check_duplicates(data: list[dict]) -> list[dict]:
    """Return list of issues for duplicate NL or SQL queries."""
    issues = []
    seen_nl  = {}
    seen_sql = {}

    for i, row in enumerate(data):
        nl  = row.get("nl_query", "").strip()
        sql = re.sub(r'\s+', ' ', row.get("sql_query", "").strip().upper())

        if nl in seen_nl:
            issues.append({
                "row": i,
                "check": "duplicate_nl",
                "detail": f"Same NL as row {seen_nl[nl]}: '{nl[:60]}'"
            })
        else:
            seen_nl[nl] = i

        if sql in seen_sql:
            issues.append({
                "row": i,
                "check": "duplicate_sql",
                "detail": f"Same SQL as row {seen_sql[sql]}"
            })
        else:
            seen_sql[sql] = i

    return issues


# -------------------------------------------------------
# Main validator
# -------------------------------------------------------

def validate(data: list[dict]) -> dict:
    """
    Run all checks and return a structured report.
    """
    all_issues = []

    # Per-row checks
    for i, row in enumerate(data):
        for check_fn in ROW_CHECKS:
            result = check_fn(row)
            if result:
                all_issues.append({
                    "row"   : i,
                    "check" : check_fn.__name__.replace("check_", ""),
                    "detail": result,
                    "nl"    : row.get("nl_query", "")[:60],
                })

    # Dataset-level duplicate checks
    all_issues.extend(check_duplicates(data))

    # Summary stats
    scenario_counts = defaultdict(int)
    source_counts   = defaultdict(int)
    diff_counts     = defaultdict(int)
    for row in data:
        scenario_counts[row.get("scenario", "?")] += 1
        source_counts[row.get("source", "?")]     += 1
        diff_counts[row.get("difficulty", "?")]   += 1

    passed = len(data) - len(set(i["row"] for i in all_issues))

    return {
        "total_pairs"       : len(data),
        "passed"            : passed,
        "failed"            : len(data) - passed,
        "total_issues"      : len(all_issues),
        "issues"            : all_issues,
        "scenario_counts"   : dict(scenario_counts),
        "source_counts"     : dict(source_counts),
        "difficulty_counts" : dict(diff_counts),
    }


# -------------------------------------------------------
# Loader (reusable by smoke_test.py and future scripts)
# -------------------------------------------------------

def load_clean_pairs(path: str | Path = INPUT_PATH) -> list[dict]:
    """
    Load pairs_clean.json and return as a list of dicts.
    Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run clean.py first."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


# -------------------------------------------------------
# Pretty printer
# -------------------------------------------------------

def print_report(report: dict) -> None:
    total  = report["total_pairs"]
    passed = report["passed"]
    failed = report["failed"]
    issues = report["issues"]

    status = "PASSED" if failed == 0 else "FAILED"

    print(f"\n{'='*55}")
    print(f"  VALIDATION REPORT  [{status}]")
    print(f"{'='*55}")
    print(f"  Total pairs   : {total}")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {failed}")
    print(f"  Total issues  : {len(issues)}")
    print(f"{'-'*55}")

    print(f"\n  Scenario breakdown:")
    for s, c in sorted(report["scenario_counts"].items()):
        bar = "#" * c
        print(f"    {s:15s} {c:3d}  {bar}")

    print(f"\n  Source breakdown:")
    for s, c in sorted(report["source_counts"].items()):
        print(f"    {s:10s} : {c}")

    print(f"\n  Difficulty breakdown:")
    for d, c in sorted(report["difficulty_counts"].items()):
        print(f"    {d:8s} : {c}")

    if issues:
        print(f"\n  Issues found:")
        print(f"  {'Row':<5} {'Check':<30} {'Detail'}")
        print(f"  {'-'*5} {'-'*30} {'-'*30}")
        for issue in issues:
            print(f"  {issue['row']:<5} {issue['check']:<30} {issue['detail'][:50]}")
    else:
        print(f"\n  No issues found. Dataset is clean!")

    print(f"{'='*55}\n")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading {INPUT_PATH} ...")
    data = load_clean_pairs()
    print(f"Loaded {len(data)} pairs\n")

    print("Running validation checks...")
    report = validate(data)

    print_report(report)

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved -> {REPORT_PATH.resolve()}")