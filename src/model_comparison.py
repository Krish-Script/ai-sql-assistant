"""
model_comparison.py -- Task 3: Compare LLaMA-3.3-70B vs LLaMA-3.1-8B on golden set
Reads  : data/golden/golden.json
Writes : outputs/model_comparison.json

Run:
    python src/model_comparison.py
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from prompt_builder import PromptBuilder
from output_parser  import OutputParser
from metrics        import score, semantic_similarity_batch

# -------------------------------------------------------
# Config
# -------------------------------------------------------

GOLDEN_PATH = Path("data/golden/golden.json")
OUTPUT_PATH = Path("outputs/model_comparison.json")

MODELS = [
    "llama-3.3-70b-versatile",   # Week 1-5 baseline
    "llama-3.1-8b-instant",      # Smaller, faster comparison
]

STRATEGY    = "zero_shot"
USE_RAG     = True               # Compare with RAG on for fairness
TEMPERATURE = 0
MAX_TOKENS  = 300
SLEEP_SEC   = 0.5


# -------------------------------------------------------
# Runner
# -------------------------------------------------------

def run_model(client: Groq, model: str, pairs: list[dict]) -> dict:
    """Run all pairs through one model and return results."""
    parser  = OutputParser()
    builder = PromptBuilder(strategy=STRATEGY)
    results = []
    latencies = []

    print(f"\n  Model: {model}")
    print(f"  {'-'*50}")

    for i, pair in enumerate(pairs, 1):
        nl       = pair["nl_query"]
        expected = pair["sql_query"]
        schema   = pair.get("schema", {})
        gold_id  = pair.get("gold_id", f"pair_{i}")

        prompt_result = builder.build(nl_query=nl, schema=schema)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model      = model,
                temperature= TEMPERATURE,
                max_tokens = MAX_TOKENS,
                messages   = [
                    {"role": "system", "content": prompt_result.system_prompt},
                    {"role": "user",   "content": prompt_result.prompt},
                ],
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            raw = ""
            print(f"    [{i}] ERROR: {str(e)[:60]}")

        latency = round(time.time() - t0, 3)
        latencies.append(latency)

        parsed      = parser.parse(raw)
        score_result = score(parsed.sql, expected)

        icon = "+" if score_result.verdict == "PASS" \
               else "~" if score_result.verdict == "PARTIAL" \
               else "x"
        print(f"    [{i:02d}] [{icon}] {score_result.verdict:7s}  "
              f"{latency:.2f}s  {nl[:45]}")

        results.append({
            "gold_id"    : gold_id,
            "nl_query"   : nl,
            "expected_sql": expected,
            "parsed_sql" : parsed.sql,
            "verdict"    : score_result.verdict,
            "exact_match": score_result.exact,
            "valid_sql"  : score_result.valid,
            "latency_s"  : latency,
        })
        time.sleep(SLEEP_SEC)

    # Compute semantic similarity batch
    pairs_for_sem = [(r["parsed_sql"], r["expected_sql"]) for r in results]
    sem_scores    = semantic_similarity_batch(pairs_for_sem)

    total   = len(results)
    passed  = sum(1 for r in results if r["verdict"] == "PASS")
    partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
    failed  = sum(1 for r in results if r["verdict"] == "FAIL")
    valid   = sum(1 for r in results if r["valid_sql"])
    avg_lat = round(sum(latencies) / len(latencies), 3) if latencies else 0
    avg_sem = round(sum(sem_scores) / len(sem_scores), 4) if sem_scores else 0

    summary = {
        "model"      : model,
        "total"      : total,
        "pass"       : passed,
        "partial"    : partial,
        "fail"       : failed,
        "pass_rate"  : round(passed / total, 3) if total else 0,
        "valid_rate" : round(valid  / total, 3) if total else 0,
        "avg_latency": avg_lat,
        "avg_semantic": avg_sem,
    }

    print(f"\n  Summary: PASS={passed}/{total} "
          f"VALID={valid}/{total} "
          f"AVG_LAT={avg_lat}s "
          f"AVG_SEM={avg_sem}")

    return {"model": model, "summary": summary, "results": results,
            "semantic_scores": sem_scores}


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found.")
        sys.exit(1)

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    test_pairs = [g for g in golden if g.get("is_correct", True)]
    print(f"Loaded {len(test_pairs)} correct golden pairs")
    print(f"Strategy : {STRATEGY}  |  RAG: {'on' if USE_RAG else 'off'}")
    print(f"Models   : {MODELS}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)
    client = Groq(api_key=api_key)

    print(f"\n{'='*55}")
    print(f"  MODEL COMPARISON")
    print(f"{'='*55}")

    model_blocks = []
    for model in MODELS:
        block = run_model(client, model, test_pairs)
        model_blocks.append(block)

    # Print comparison table
    print(f"\n{'='*65}")
    print(f"  MODEL COMPARISON TABLE  (strategy={STRATEGY}, n={len(test_pairs)})")
    print(f"{'='*65}")
    print(f"  {'Model':<30} {'PASS':>6} {'VALID':>6} {'AVG LAT':>8} {'AVG SEM':>8}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for block in model_blocks:
        s = block["summary"]
        n = s["total"]
        print(f"  {s['model']:<30} "
              f"{s['pass']:>3}/{n}  "
              f"{s['valid_rate']*100:>4.0f}%  "
              f"{s['avg_latency']:>7.2f}s  "
              f"{s['avg_semantic']:>8.4f}")
    print(f"{'='*65}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "run_date" : datetime.now().isoformat(),
        "strategy" : STRATEGY,
        "use_rag"  : USE_RAG,
        "n_pairs"  : len(test_pairs),
        "models"   : model_blocks,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {OUTPUT_PATH.resolve()}")
    print("Task 3 complete!")


if __name__ == "__main__":
    main()