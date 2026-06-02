"""
smoke_test.py -- Task 7: Run 5 pairs through Groq + LLaMA-3 as a smoke test
Reads  : data/golden/golden.json  (uses the golden set for meaningful evaluation)
Writes : outputs/smoke_test_results.json

What this does:
    1. Loads the golden set
    2. Picks 5 pairs (skips the intentionally wrong one)
    3. Sends each NL query to LLaMA-3 via Groq
    4. Compares generated SQL vs expected SQL
    5. Scores each result and saves a report

Comparison strategy:
    - Exact match         : generated SQL == expected SQL (normalized)
    - Keyword match       : key SQL keywords present (SELECT, FROM, WHERE etc.)
    - Table match         : correct table names mentioned
    - Overall verdict     : PASS / PARTIAL / FAIL
"""

import json
import re
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq # type: ignore

load_dotenv()

# -------------------------------------------------------
# Config
# -------------------------------------------------------

GOLDEN_PATH = Path("data/golden/golden.json")
OUTPUT_PATH = Path("outputs/smoke_test_results.json")
MODEL       = "llama-3.3-70b-versatile"   # Best free LLaMA-3 model on Groq
N_PAIRS     = 5                   # Number of pairs to test


# -------------------------------------------------------
# Groq client
# -------------------------------------------------------

def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file.")
        print("Make sure your .env file contains: GROQ_API_KEY=your-key-here")
        sys.exit(1)
    return Groq(api_key=api_key)


# -------------------------------------------------------
# SQL generator
# -------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL assistant. 
Given a natural language question and a database schema, return ONLY the SQL query.
Do not explain. Do not use markdown. Do not add any text before or after the SQL.
The SQL must be valid and executable."""

def generate_sql(client: Groq, nl_query: str, schema: dict) -> str:
    """Send NL query + schema to LLaMA-3 and return generated SQL."""

    # Format schema for the prompt
    tables  = ", ".join(schema.get("tables", []))
    columns = ", ".join(schema.get("columns", [])[:15])  # cap at 15 cols
    schema_text = f"Tables: {tables}\nColumns: {columns}"

    user_prompt = f"""Database schema:
{schema_text}

Question: {nl_query}

SQL:"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,        # deterministic output
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------------
# Comparator
# -------------------------------------------------------

def normalize(sql: str) -> str:
    """Normalize SQL for comparison — uppercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', sql.strip().upper())


def extract_keywords(sql: str) -> set[str]:
    """Extract SQL keywords present in a query."""
    keywords = [
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY",
        "HAVING", "LIMIT", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "DISTINCT", "IN", "NOT IN", "EXISTS", "UNION", "EXCEPT",
    ]
    sql_upper = sql.upper()
    return {kw for kw in keywords if kw in sql_upper}


def extract_tables(sql: str, schema: dict) -> set[str]:
    """Check which schema tables appear in the generated SQL."""
    sql_upper = sql.upper()
    return {t for t in schema.get("tables", []) if t.upper() in sql_upper}


def compare(generated: str, expected: str, schema: dict) -> dict:
    """
    Compare generated vs expected SQL.
    Returns a dict with scores and verdict.
    """
    gen_norm = normalize(generated)
    exp_norm = normalize(expected)

    # Exact match
    exact = gen_norm == exp_norm

    # Keyword overlap
    gen_kw  = extract_keywords(generated)
    exp_kw  = extract_keywords(expected)
    kw_overlap = len(gen_kw & exp_kw)
    kw_total   = len(exp_kw)
    kw_score   = round(kw_overlap / kw_total, 2) if kw_total > 0 else 0.0

    # Table match
    expected_tables = extract_tables(expected, schema)
    generated_tables = extract_tables(generated, schema)
    table_match = expected_tables == generated_tables

    # Verdict
    if exact:
        verdict = "PASS"
    elif kw_score >= 0.75 and table_match:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return {
        "exact_match"  : exact,
        "keyword_score": kw_score,
        "table_match"  : table_match,
        "verdict"      : verdict,
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    # Load golden set
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found. Run golden.py first.")
        sys.exit(1)

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    # Only test correct pairs (skip the intentional wrong one)
    correct_pairs = [g for g in golden if g.get("is_correct", True)]
    test_pairs    = correct_pairs[:N_PAIRS]

    print(f"Loaded {len(golden)} golden pairs")
    print(f"Testing {len(test_pairs)} correct pairs with {MODEL} via Groq\n")
    print("=" * 60)

    client  = get_client()
    results = []

    for i, pair in enumerate(test_pairs, start=1):
        gold_id  = pair.get("gold_id", f"pair_{i}")
        nl       = pair["nl_query"]
        expected = pair["sql_query"]
        schema   = pair.get("schema", {})
        scenario = pair.get("scenario", "")
        diff     = pair.get("difficulty", "")

        print(f"\n[{i}/{N_PAIRS}] {gold_id}  [{scenario}]  [{diff}]")
        print(f"  NL       : {nl}")
        print(f"  Expected : {expected}")

        # Generate
        try:
            generated = generate_sql(client, nl, schema)
        except Exception as e:
            generated = f"ERROR: {e}"

        print(f"  Generated: {generated}")

        # Compare
        scores  = compare(generated, expected, schema)
        verdict = scores["verdict"]
        icon    = "+" if verdict == "PASS" else "~" if verdict == "PARTIAL" else "x"

        print(f"  Result   : [{icon}] {verdict}"
              f"  (keywords={scores['keyword_score']}"
              f"  tables={scores['table_match']}"
              f"  exact={scores['exact_match']})")

        results.append({
            "gold_id"       : gold_id,
            "scenario"      : scenario,
            "difficulty"    : diff,
            "nl_query"      : nl,
            "expected_sql"  : expected,
            "generated_sql" : generated,
            "exact_match"   : scores["exact_match"],
            "keyword_score" : scores["keyword_score"],
            "table_match"   : scores["table_match"],
            "verdict"       : verdict,
        })

    # Summary
    passed   = sum(1 for r in results if r["verdict"] == "PASS")
    partial  = sum(1 for r in results if r["verdict"] == "PARTIAL")
    failed   = sum(1 for r in results if r["verdict"] == "FAIL")

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY  ({MODEL})")
    print(f"{'='*60}")
    print(f"  Total tested : {len(results)}")
    print(f"  PASS         : {passed}")
    print(f"  PARTIAL      : {partial}")
    print(f"  FAIL         : {failed}")
    avg_kw = round(sum(r["keyword_score"] for r in results) / len(results), 2)
    print(f"  Avg keyword  : {avg_kw}")
    print(f"{'='*60}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model"  : MODEL,
        "tested" : len(results),
        "passed" : passed,
        "partial": partial,
        "failed" : failed,
        "avg_keyword_score": avg_kw,
        "results": results,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved -> {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()