"""
eval_runner.py -- Task 3: Runs NL->SQL pairs through a prompt strategy + model
Wires together: PromptBuilder -> Groq API -> OutputParser -> Metrics

Used by: Task 5 experiment (3x1 grid: 3 strategies x 1 model)
Writes : outputs/results_week2.json

Pipeline per pair:
    1. PromptBuilder  builds prompt from strategy
    2. Groq API       generates raw SQL
    3. OutputParser   cleans raw response
    4. Metrics        scores cleaned SQL vs expected
    5. Result         logged with full metadata

Usage:
    # Run single strategy
    python src/eval_runner.py --strategy zero_shot

    # Run all 3 strategies (full 3x1 experiment)
    python src/eval_runner.py --all

    # Run on golden set only (faster, for quick checks)
    python src/eval_runner.py --strategy few_shot --golden-only
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from prompt_builder import PromptBuilder, VALID_STRATEGIES
from output_parser  import OutputParser
from metrics        import score, summarize, semantic_similarity_batch, summarize_with_semantics
from retriever      import Retriever

# -------------------------------------------------------
# Config
# -------------------------------------------------------

GOLDEN_PATH = Path("data/golden/golden.json")
PAIRS_PATH  = Path("data/processed/pairs_clean.json")
OUTPUT_PATH          = Path("outputs/results_week4.json")
OUTPUT_PATH_WEEK5    = Path("outputs/results_week5_rag_full.json")

MODEL       = "llama-3.3-70b-versatile"
SLEEP_SEC   = 0.5      # Groq free tier rate limit buffer
TEMPERATURE = 0        # Always 0 for reproducibility
MAX_TOKENS  = 300


# -------------------------------------------------------
# Groq client
# -------------------------------------------------------

def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file.")
        sys.exit(1)
    return Groq(api_key=api_key)


# -------------------------------------------------------
# Single pair runner
# -------------------------------------------------------

def run_pair(
    client  : Groq,
    builder : PromptBuilder,
    parser  : OutputParser,
    pair    : dict,
    idx     : int,
    total   : int,
    dry_run : bool = False,
) -> dict:
    """
    Run a single NL->SQL pair through the full pipeline.
    Returns a result dict with all metadata.
    """
    nl       = pair["nl_query"]
    expected = pair["sql_query"]
    schema   = pair.get("schema", {})
    gold_id  = pair.get("gold_id", f"pair_{idx}")
    scenario = pair.get("scenario", "")
    diff     = pair.get("difficulty", "")

    print(f"  [{idx}/{total}] {gold_id}  [{scenario}]  [{diff}]")
    print(f"    NL: {nl[:70]}")

    # Step 1: Build prompt
    prompt_result = builder.build(nl_query=nl, schema=schema)

    # Step 2: Call Groq API (skipped in dry-run mode)
    raw_output = ""
    api_error  = ""
    if dry_run:
        raw_output = f"[DRY RUN] SELECT * FROM dry_run_table"
        print(f"    [DRY RUN] Skipping LLM call")
    else:
     try:
        response = client.chat.completions.create(
            model      = MODEL,
            temperature= TEMPERATURE,
            max_tokens = MAX_TOKENS,
            messages   = [
                {"role": "system", "content": prompt_result.system_prompt},
                {"role": "user",   "content": prompt_result.prompt},
            ],
        )
        raw_output = response.choices[0].message.content.strip()
     except Exception as e:
        api_error  = str(e)
        raw_output = ""
        print(f"    API ERROR: {api_error[:80]}")

    # Step 3: Parse output
    parse_result = parser.parse(raw_output)

    # Step 4: Score
    score_result = score(parse_result.sql, expected)

    # Print result
    icon = "+" if score_result.verdict == "PASS" \
           else "~" if score_result.verdict == "PARTIAL" \
           else "x"
    print(f"    Expected : {expected[:70]}")
    print(f"    Generated: {parse_result.sql[:70]}")
    print(f"    Result   : [{icon}] {score_result.verdict}"
          f"  exact={score_result.exact}"
          f"  valid={score_result.valid}")

    # Rate limit buffer
    time.sleep(SLEEP_SEC)

    return {
        "gold_id"       : gold_id,
        "scenario"      : scenario,
        "difficulty"    : diff,
        "nl_query"      : nl,
        "expected_sql"  : expected,
        "raw_output"    : raw_output,
        "parsed_sql"    : parse_result.sql,
        "parse_success" : parse_result.success,
        "parse_issues"  : parse_result.issues,
        "verdict"       : score_result.verdict,
        "exact_match"   : score_result.exact,
        "valid_sql"     : score_result.valid,
        "score_notes"   : score_result.notes,
        "api_error"     : api_error,
        "strategy"      : builder.strategy,
        "model"         : MODEL,
        "examples_used" : prompt_result.examples_used,
    }


# -------------------------------------------------------
# Strategy runner
# -------------------------------------------------------

def run_strategy(
    client      : Groq,
    strategy    : str,
    pairs       : list[dict],
    golden_only : bool = False,
    dry_run     : bool = False,
) -> dict:
    """
    Run all pairs through one strategy and return a strategy result block.
    """
    # Filter to correct pairs only (skip intentional wrong pair)
    test_pairs = [p for p in pairs if p.get("is_correct", True)]

    if golden_only:
        source = "golden"
    else:
        source = "golden"   # Week 2: always use golden set for fair comparison

    print(f"\n{'='*60}")
    print(f"  Strategy : {strategy.upper()}")
    print(f"  Model    : {MODEL}")
    print(f"  Pairs    : {len(test_pairs)}")
    print(f"{'='*60}")

    builder = PromptBuilder(strategy=strategy)
    parser  = OutputParser()
    results = []

    for i, pair in enumerate(test_pairs, start=1):
        result = run_pair(client, builder, parser, pair, i, len(test_pairs), dry_run=dry_run)
        results.append(result)

    summary = summarize([type("R", (), {"verdict": r["verdict"]})() for r in results]) # pyright: ignore[reportArgumentType]

    # Fix summarize compatibility -- pass proper ScoreResult-like objects
    pass_c    = sum(1 for r in results if r["verdict"] == "PASS")
    partial_c = sum(1 for r in results if r["verdict"] == "PARTIAL")
    fail_c    = sum(1 for r in results if r["verdict"] == "FAIL")
    total     = len(results)

    strategy_summary = {
        "total"       : total,
        "pass"        : pass_c,
        "partial"     : partial_c,
        "fail"        : fail_c,
        "pass_rate"   : round(pass_c    / total, 3) if total else 0,
        "partial_rate": round(partial_c / total, 3) if total else 0,
        "fail_rate"   : round(fail_c    / total, 3) if total else 0,
        "valid_rate"  : round((pass_c + partial_c) / total, 3) if total else 0,
    }

    print(f"\n  [{strategy.upper()}] Summary:")
    print(f"    PASS    : {pass_c}/{total}  ({strategy_summary['pass_rate']*100:.0f}%)")
    print(f"    PARTIAL : {partial_c}/{total}  ({strategy_summary['partial_rate']*100:.0f}%)")
    print(f"    FAIL    : {fail_c}/{total}  ({strategy_summary['fail_rate']*100:.0f}%)")
    print(f"    VALID   : {strategy_summary['valid_rate']*100:.0f}%")

    return {
        "strategy": strategy,
        "model"   : MODEL,
        "summary" : strategy_summary,
        "results" : results,
    }


# -------------------------------------------------------
# Comparison table printer
# -------------------------------------------------------

def print_comparison_table(strategy_blocks: list[dict]) -> None:
    """Print a side-by-side comparison of all strategies."""
    print(f"\n{'='*60}")
    print(f"  3x1 EXPERIMENT RESULTS  ({MODEL})")
    print(f"{'='*60}")
    print(f"  {'Strategy':<20} {'PASS':>6} {'PARTIAL':>8} {'FAIL':>6} {'VALID':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*6} {'-'*7}")

    for block in strategy_blocks:
        s  = block["summary"]
        n  = s["total"]
        print(
            f"  {block['strategy']:<20} "
            f"{s['pass']:>3}/{n}  "
            f"{s['partial']:>4}/{n}  "
            f"{s['fail']:>3}/{n}  "
            f"{s['valid_rate']*100:>5.0f}%"
        )

    print(f"{'='*60}")

    # Find winner
    best = max(strategy_blocks,
               key=lambda b: (b["summary"]["pass"], b["summary"]["valid_rate"]))
    print(f"\n  Best strategy: {best['strategy'].upper()}"
          f"  (PASS={best['summary']['pass']}, "
          f"VALID={best['summary']['valid_rate']*100:.0f}%)")
    print(f"{'='*60}\n")


# -------------------------------------------------------
# RAG vs Baseline comparison (Task 8)
# -------------------------------------------------------

def run_rag_comparison(client: Groq, pairs: list[dict]) -> dict:
    """
    Run RAG vs baseline comparison on a set of pairs.
    Returns a dict with both condition results and a comparison summary.
    """
    from retriever import Retriever

    test_pairs = [p for p in pairs if p.get("is_correct", True)]
    print(f"RAG vs Baseline comparison on {len(test_pairs)} pairs")

    retriever = Retriever()
    parser    = OutputParser()

    baseline_results = []
    rag_results      = []

    for i, pair in enumerate(test_pairs, 1):
        nl       = pair["nl_query"]
        expected = pair["sql_query"]
        schema   = pair.get("schema", {})
        gold_id  = pair.get("gold_id", f"pair_{i}")
        scenario = pair.get("scenario", "")

        print(f"  [{i}/{len(test_pairs)}] {gold_id}  [{scenario}]")

        # -- Baseline: zero_shot no RAG --
        baseline_builder = PromptBuilder(strategy="zero_shot")
        baseline_prompt  = baseline_builder.build(nl_query=nl, schema=schema)
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": baseline_prompt.system_prompt},
                    {"role": "user",   "content": baseline_prompt.prompt},
                ],
            )
            baseline_raw = resp.choices[0].message.content.strip()
        except Exception as e:
            baseline_raw = ""
        baseline_parsed = parser.parse(baseline_raw)
        baseline_score  = score(baseline_parsed.sql, expected)
        time.sleep(SLEEP_SEC)

        # -- RAG: few_shot with retrieved examples --
        retrieved    = retriever.retrieve(nl, k=3)
        rag_examples = [
            {"nl_query": r.nl_query, "sql_query": r.sql_query, "schema": {}}
            for r in retrieved
        ]
        rag_builder             = PromptBuilder(strategy="few_shot")
        rag_builder._safe_pairs = rag_examples
        rag_prompt              = rag_builder.build(nl_query=nl, schema=schema)
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": rag_prompt.system_prompt},
                    {"role": "user",   "content": rag_prompt.prompt},
                ],
            )
            rag_raw = resp.choices[0].message.content.strip()
        except Exception as e:
            rag_raw = ""
        rag_parsed = parser.parse(rag_raw)
        rag_score  = score(rag_parsed.sql, expected)
        time.sleep(SLEEP_SEC)

        b_icon = "+" if baseline_score.verdict == "PASS" else "~" if baseline_score.verdict == "PARTIAL" else "x"
        r_icon = "+" if rag_score.verdict == "PASS" else "~" if rag_score.verdict == "PARTIAL" else "x"
        print(f"    Baseline [{b_icon}] {baseline_score.verdict}  |  RAG [{r_icon}] {rag_score.verdict}")

        baseline_results.append({
            "gold_id": gold_id, "scenario": scenario,
            "nl_query": nl, "expected_sql": expected,
            "parsed_sql": baseline_parsed.sql,
            "verdict": baseline_score.verdict,
            "exact_match": baseline_score.exact,
            "valid_sql": baseline_score.valid,
        })
        rag_results.append({
            "gold_id": gold_id, "scenario": scenario,
            "nl_query": nl, "expected_sql": expected,
            "parsed_sql": rag_parsed.sql,
            "verdict": rag_score.verdict,
            "exact_match": rag_score.exact,
            "valid_sql": rag_score.valid,
            "retrieved_similarities": [r.similarity for r in retrieved],
        })

    # Compute semantic similarity for both conditions
    baseline_pairs = [(r["parsed_sql"], r["expected_sql"]) for r in baseline_results]
    rag_pairs      = [(r["parsed_sql"], r["expected_sql"]) for r in rag_results]

    baseline_sem = semantic_similarity_batch(baseline_pairs)
    rag_sem      = semantic_similarity_batch(rag_pairs)

    def _summary(results, sem_scores):
        n       = len(results)
        passed  = sum(1 for r in results if r["verdict"] == "PASS")
        partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
        failed  = sum(1 for r in results if r["verdict"] == "FAIL")
        return {
            "total": n, "pass": passed, "partial": partial, "fail": failed,
            "pass_rate"   : round(passed / n, 3) if n else 0,
            "valid_rate"  : round((passed + partial) / n, 3) if n else 0,
            "avg_semantic": round(sum(sem_scores) / len(sem_scores), 4) if sem_scores else 0,
        }

    baseline_summary = _summary(baseline_results, baseline_sem)
    rag_summary      = _summary(rag_results, rag_sem)

    # Print comparison table
    print(f"{'='*60}")
    print(f"  RAG vs BASELINE COMPARISON  ({MODEL})")
    print(f"{'='*60}")
    print(f"  {'Condition':<15} {'PASS':>6} {'VALID':>7} {'AVG SEM':>9}")
    print(f"  {'-'*15} {'-'*6} {'-'*7} {'-'*9}")
    n = baseline_summary["total"]
    print(f"  {'Baseline':<15} {baseline_summary['pass']:>3}/{n}  "
          f"{baseline_summary['valid_rate']*100:>5.0f}%  "
          f"{baseline_summary['avg_semantic']:>9.4f}")
    print(f"  {'RAG':<15} {rag_summary['pass']:>3}/{n}  "
          f"{rag_summary['valid_rate']*100:>5.0f}%  "
          f"{rag_summary['avg_semantic']:>9.4f}")
    print(f"{'='*60}")

    sem_delta  = round(rag_summary["avg_semantic"] - baseline_summary["avg_semantic"], 4)
    pass_delta = rag_summary["pass"] - baseline_summary["pass"]
    print(f"Semantic similarity delta: {sem_delta:+.4f}")
    print(f"  PASS count delta         : {pass_delta:+d}")
    print(f"{'='*60}")

    return {
        "baseline": {"summary": baseline_summary, "results": baseline_results,
                     "semantic_scores": baseline_sem},
        "rag"     : {"summary": rag_summary,      "results": rag_results,
                     "semantic_scores": rag_sem},
        "deltas"  : {"semantic": sem_delta, "pass_count": pass_delta},
    }


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser_arg = argparse.ArgumentParser(
        description="Run NL->SQL eval across prompt strategies"
    )
    parser_arg.add_argument(
        "--strategy", choices=VALID_STRATEGIES,
        help="Run a single strategy"
    )
    parser_arg.add_argument(
        "--all", action="store_true",
        help="Run all 3 strategies (full 3x1 experiment)"
    )
    parser_arg.add_argument(
        "--golden-only", action="store_true", default=False,
        help="Evaluate on golden set only"
    )
    parser_arg.add_argument(
        "--full-pairs", action="store_true", default=False,
        help="Evaluate on all 64 pairs from pairs_clean.json (Task 5)"
    )
    parser_arg.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Skip LLM calls, test pipeline only (no API usage)"
    )
    parser_arg.add_argument(
        "--rag", action="store_true", default=False,
        help="Run RAG vs baseline comparison on golden set (Task 8)"
    )
    args = parser_arg.parse_args()

    # Default to --all if nothing specified
    if not args.strategy and not args.all:
        args.all = True

    # Load pairs
    if args.full_pairs:
        # Task 5: full 64-pair evaluation
        if not PAIRS_PATH.exists():
            print(f"ERROR: {PAIRS_PATH} not found. Run clean.py first.")
            sys.exit(1)
        with open(PAIRS_PATH, encoding="utf-8") as f:
            pairs = json.load(f)
        # Add is_correct=True for all clean pairs (no intentional wrong pairs)
        for p in pairs:
            p.setdefault("is_correct", True)
            p.setdefault("gold_id", p.get("nl_query", "")[:20])
        print(f"Loaded {len(pairs)} pairs from pairs_clean.json")
    else:
        # Default: golden set evaluation
        if not GOLDEN_PATH.exists():
            print(f"ERROR: {GOLDEN_PATH} not found. Run golden.py first.")
            sys.exit(1)
        with open(GOLDEN_PATH, encoding="utf-8") as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} golden pairs")

    client = get_client()

    # Determine which strategies to run
    # RAG comparison mode
    if args.rag:
        comparison = run_rag_comparison(client=client, pairs=pairs)
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "run_date"  : datetime.now().isoformat(),
            "model"     : MODEL,
            "run_type"  : "rag_comparison",
            "comparison": comparison,
        }
        out_path = OUTPUT_PATH_WEEK5 if args.full_pairs else OUTPUT_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Results saved -> {out_path.resolve()}")
        print("RAG comparison complete!")
        return

    strategies = list(VALID_STRATEGIES) if args.all else [args.strategy]

    strategy_blocks = []
    for strategy in strategies:
        block = run_strategy(
            client      = client,
            strategy    = strategy,
            pairs       = pairs,
            golden_only = args.golden_only,
            dry_run     = args.dry_run,
        )
        strategy_blocks.append(block)

    # Print comparison table
    if len(strategy_blocks) > 1:
        print_comparison_table(strategy_blocks)

    # Save full results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "run_date"  : datetime.now().isoformat(),
        "model"     : MODEL,
        "strategies": strategy_blocks,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Results saved -> {OUTPUT_PATH.resolve()}")
    print("Eval complete!")


if __name__ == "__main__":
    main()