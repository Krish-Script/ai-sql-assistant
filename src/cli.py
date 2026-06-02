"""
cli.py -- Task 6: CLI client for the AI SQL Query Assistant API
Hits the Flask API at localhost:5000 and prints results to terminal.

Usage:
    # Interactive mode (default)
    python src/cli.py

    # Single query mode
    python src/cli.py --query "How many airports do we have?"

    # With strategy
    python src/cli.py --query "Count airports by country" --strategy few_shot

    # With schema
    python src/cli.py --query "Count airports by country" \
        --tables airports flights \
        --columns airport_id city country

    # Evaluate a pair
    python src/cli.py --evaluate \
        --query "How many airports?" \
        --expected "SELECT COUNT(*) FROM airports"

    # Health check
    python src/cli.py --health
"""

import sys
import json
import argparse
import requests
from typing import Optional

# -------------------------------------------------------
# Config
# -------------------------------------------------------

API_BASE    = "http://localhost:5000"
TIMEOUT_SEC = 30    # LLM calls can take 3-8s, give plenty of headroom


# -------------------------------------------------------
# API client functions
# -------------------------------------------------------

def check_health() -> bool:
    """Check if the API server is running. Returns True if healthy."""
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Server  : {API_BASE}")
            print(f"  Status  : {data['status'].upper()}")
            print(f"  Model   : {data['model']}")
            print(f"  Strategies: {', '.join(data['strategies'])}")
            return True
        else:
            print(f"  Server returned {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to {API_BASE}")
        print(f"  Make sure the API is running: python src/api.py")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def generate_sql(
    nl_query : str,
    schema   : Optional[dict] = None,
    strategy : str = "zero_shot",
    use_rag  : bool = False,
) -> Optional[dict]:
    """
    Call POST /generate and return the response dict.
    Returns None on failure.
    """
    payload = {
        "nl_query": nl_query,
        "strategy": strategy,
        "schema"  : schema or {},
        "use_rag" : use_rag,
    }

    try:
        print("Generating...", end="", flush=True)
        resp = requests.post(
            f"{API_BASE}/generate",
            json   = payload,
            timeout= TIMEOUT_SEC,
        )
        print("\r", end="")  # clear "Generating..." line

        if resp.status_code == 200:
            return resp.json()
        else:
            error = resp.json().get("error", "Unknown error")
            print(f"  API Error ({resp.status_code}): {error}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"\r  ERROR: Cannot connect to {API_BASE}")
        print(f"  Make sure the API is running: python src/api.py")
        return None
    except requests.exceptions.Timeout:
        print(f"\r  ERROR: Request timed out after {TIMEOUT_SEC}s")
        return None
    except Exception as e:
        print(f"\r  ERROR: {e}")
        return None


def evaluate_pair(
    nl_query     : str,
    expected_sql : str,
    schema       : Optional[dict] = None,
    strategy     : str = "zero_shot",
) -> Optional[dict]:
    """
    Call POST /evaluate and return the response dict.
    Returns None on failure.
    """
    payload = {
        "nl_query"    : nl_query,
        "expected_sql": expected_sql,
        "strategy"    : strategy,
        "schema"      : schema or {},
    }

    try:
        print("Generating and evaluating...", end="", flush=True)
        resp = requests.post(
            f"{API_BASE}/evaluate",
            json   = payload,
            timeout= TIMEOUT_SEC,
        )
        print("\r", end="")

        if resp.status_code == 200:
            return resp.json()
        else:
            error = resp.json().get("error", "Unknown error")
            print(f"  API Error ({resp.status_code}): {error}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"\r  ERROR: Cannot connect to {API_BASE}")
        return None
    except Exception as e:
        print(f"\r  ERROR: {e}")
        return None


# -------------------------------------------------------
# Display helpers
# -------------------------------------------------------

def print_generate_result(result: dict, nl_query: str) -> None:
    """Pretty print a /generate response."""
    valid_icon = "+" if result.get("valid") else "x"
    print()
    print(f"  Question : {nl_query}")
    print(f"  Strategy : {result.get('strategy', '?')}")
    print(f"  Model    : {result.get('model', '?')}")
    print(f"  Latency  : {result.get('latency_s', '?')}s")
    print(f"  Valid SQL : [{valid_icon}] {result.get('valid')}")
    if result.get("parse_issues"):
        print(f"  Fixed    : {result['parse_issues']}")
    print()
    print(f"  SQL:")
    print(f"  {'-'*55}")
    print(f"  {result.get('sql', '(empty)')}")
    print(f"  {'-'*55}")


def print_evaluate_result(result: dict) -> None:
    """Pretty print a /evaluate response."""
    verdict = result.get("verdict", "?")
    icon    = "+" if verdict == "PASS" else "~" if verdict == "PARTIAL" else "x"
    print()
    print(f"  Question   : {result.get('nl_query', '?')}")
    print(f"  Strategy   : {result.get('strategy', '?')}")
    print(f"  Latency    : {result.get('latency_s', 'N/A')}s")
    print()
    print(f"  Expected   : {result.get('expected_sql', '?')}")
    print(f"  Generated  : {result.get('generated_sql', '?')}")
    print()
    print(f"  Verdict    : [{icon}] {verdict}")
    print(f"  Exact match: {result.get('exact_match', '?')}")
    print(f"  Valid SQL  : {result.get('valid_sql', '?')}")
    if result.get("score_notes"):
        print(f"  Notes      : {result['score_notes']}")


# -------------------------------------------------------
# Interactive mode
# -------------------------------------------------------

def interactive_mode(strategy: str = "zero_shot") -> None:
    """
    Run an interactive REPL loop.
    User types NL queries and gets SQL back.
    Type 'exit' or 'quit' to stop.
    Type 'strategy <name>' to switch strategies.
    Type 'health' to check server status.
    """
    print()
    print("=" * 55)
    print("  AI SQL Query Assistant")
    print(f"  Strategy: {strategy}  |  API: {API_BASE}")
    print("  Commands: 'exit' to quit, 'health' to check server")
    print(f"            'strategy <name>' to switch strategy")
    print("=" * 55)

    # Quick health check on startup
    print("\nChecking server...")
    if not check_health():
        print("\nServer not available. Exiting.")
        sys.exit(1)

    current_strategy = strategy
    print(f"\nReady! Using strategy: {current_strategy}\n")

    while True:
        try:
            user_input = input("  NL Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "health":
            print()
            check_health()
            print()
            continue

        if user_input.lower().startswith("strategy "):
            new_strategy = user_input.split(" ", 1)[1].strip()
            valid = ["zero_shot", "few_shot", "chain_of_thought"]
            if new_strategy in valid:
                current_strategy = new_strategy
                print(f"  Switched to strategy: {current_strategy}\n")
            else:
                print(f"  Invalid strategy. Choose from: {valid}\n")
            continue

        # Generate SQL
        result = generate_sql(
            nl_query = user_input,
            strategy = current_strategy,
            use_rag  = use_rag,
        )
        if result:
            print_generate_result(result, user_input)
            if use_rag and result.get("retrieved_pairs"):
                print("  Retrieved examples:")
                for i, pair in enumerate(result["retrieved_pairs"], 1):
                    print(f"    [{i}] sim={pair['similarity']:.4f}  {pair['nl_query'][:55]}")
                    print(f"        SQL: {pair['sql_query'][:60]}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CLI client for the AI SQL Query Assistant API"
    )

    # Mode flags
    parser.add_argument("--query",    "-q", type=str, help="NL query to generate SQL for")
    parser.add_argument("--evaluate", "-e", action="store_true", help="Evaluate mode")
    parser.add_argument("--health",         action="store_true", help="Check server health")

    # Options
    parser.add_argument("--expected",  type=str, help="Expected SQL (for --evaluate)")
    parser.add_argument("--strategy",  type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "chain_of_thought"],
                        help="Prompt strategy (default: zero_shot)")
    parser.add_argument("--rag", action="store_true", default=False,
                        help="Enable RAG -- retrieves similar examples to guide generation")
    parser.add_argument("--tables",    nargs="*", default=[], help="Table names for schema")
    parser.add_argument("--columns",   nargs="*", default=[], help="Column names for schema")

    args = parser.parse_args()

    # Build schema from args
    schema = {}
    if args.tables or args.columns:
        schema = {"tables": args.tables, "columns": args.columns}

    # Health check
    if args.health:
        print("\nHealth check:")
        check_health()
        return

    # Evaluate mode
    if args.evaluate:
        if not args.query:
            print("ERROR: --query is required for --evaluate mode")
            sys.exit(1)
        if not args.expected:
            print("ERROR: --expected is required for --evaluate mode")
            sys.exit(1)
        result = evaluate_pair(
            nl_query    = args.query,
            expected_sql= args.expected,
            schema      = schema,
            strategy    = args.strategy,
        )
        if result:
            print_evaluate_result(result)
        return

    # Single query mode
    if args.query:
        result = generate_sql(
            nl_query = args.query,
            schema   = schema,
            strategy = args.strategy,
            use_rag  = args.rag,
        )
        if result:
            print_generate_result(result, args.query)
            if args.rag and result.get("retrieved_pairs"):
                print("  Retrieved examples:")
                for i, pair in enumerate(result["retrieved_pairs"], 1):
                    print(f"    [{i}] sim={pair['similarity']:.4f}  {pair['nl_query'][:55]}")
                    print(f"        SQL: {pair['sql_query'][:60]}")
        return

    # Default: interactive mode
    interactive_mode(strategy=args.strategy)


if __name__ == "__main__":
    main()