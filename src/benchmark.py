"""
benchmark.py -- Task 3: Latency benchmark across strategies + RAG overhead
Reads  : data/golden/golden.json (5 pairs used for benchmark)
Writes : outputs/benchmark_results.json
         outputs/benchmark_chart.png

Run:
    python src/benchmark.py

Measures:
    - Mean and P95 latency for each strategy (zero_shot, few_shot, chain_of_thought)
    - RAG overhead (zero_shot with RAG vs without)
    - 5 runs per condition for stable estimates
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from prompt_builder import PromptBuilder, VALID_STRATEGIES
from output_parser  import OutputParser
from retriever      import Retriever

# -------------------------------------------------------
# Config
# -------------------------------------------------------

GOLDEN_PATH  = Path("data/golden/golden.json")
OUTPUT_PATH  = Path("outputs/benchmark_results.json")
CHART_PATH   = Path("outputs/benchmark_chart.png")

MODEL        = "llama-3.3-70b-versatile"
TEMPERATURE  = 0
MAX_TOKENS   = 300
N_RUNS       = 5      # runs per condition
SLEEP_SEC    = 0.3    # smaller sleep for benchmark speed

CONDITIONS = [
    ("zero_shot",        False),
    ("few_shot",         False),
    ("chain_of_thought", False),
    ("zero_shot",        True),    # RAG condition
]

CONDITION_LABELS = {
    ("zero_shot",        False): "Zero-Shot",
    ("few_shot",         False): "Few-Shot",
    ("chain_of_thought", False): "Chain-of-Thought",
    ("zero_shot",        True) : "Zero-Shot + RAG",
}


# -------------------------------------------------------
# Single timed call
# -------------------------------------------------------

def timed_call(
    client   : Groq,
    builder  : PromptBuilder,
    parser   : OutputParser,
    pair     : dict,
    retriever: Retriever | None = None,
    use_rag  : bool = False,
) -> float:
    """Make one LLM call and return latency in seconds."""
    nl     = pair["nl_query"]
    schema = pair.get("schema", {})

    if use_rag and retriever:
        retrieved    = retriever.retrieve(nl, k=3)
        rag_examples = [
            {"nl_query": r.nl_query, "sql_query": r.sql_query, "schema": {}}
            for r in retrieved
        ]
        builder._safe_pairs = rag_examples

    prompt_result = builder.build(nl_query=nl, schema=schema)

    t0 = time.time()
    try:
        client.chat.completions.create(
            model       = MODEL,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            messages    = [
                {"role": "system", "content": prompt_result.system_prompt},
                {"role": "user",   "content": prompt_result.prompt},
            ],
        )
    except Exception as e:
        print(f"    API error: {str(e)[:60]}")
        return -1.0

    return round(time.time() - t0, 3)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    if not GOLDEN_PATH.exists():
        print(f"ERROR: {GOLDEN_PATH} not found.")
        sys.exit(1)

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = json.load(f)

    # Use first N_RUNS correct pairs
    test_pairs = [g for g in golden if g.get("is_correct", True)][:N_RUNS]
    print(f"Benchmark: {len(test_pairs)} pairs x {len(CONDITIONS)} conditions")
    print(f"Model: {MODEL}\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    client    = Groq(api_key=api_key)
    parser    = OutputParser()
    retriever = Retriever()

    results = {}

    for strategy, use_rag in CONDITIONS:
        label   = CONDITION_LABELS[(strategy, use_rag)]
        builder = PromptBuilder(strategy=strategy)
        latencies = []

        print(f"  [{label}]")
        for i, pair in enumerate(test_pairs, 1):
            lat = timed_call(
                client    = client,
                builder   = builder,
                parser    = parser,
                pair      = pair,
                retriever = retriever if use_rag else None,
                use_rag   = use_rag,
            )
            if lat >= 0:
                latencies.append(lat)
                print(f"    run {i}: {lat:.3f}s")
            time.sleep(SLEEP_SEC)

        if latencies:
            mean_lat = round(float(np.mean(latencies)), 3)
            p95_lat  = round(float(np.percentile(latencies, 95)), 3)
            min_lat  = round(float(np.min(latencies)), 3)
            max_lat  = round(float(np.max(latencies)), 3)
        else:
            mean_lat = p95_lat = min_lat = max_lat = -1.0

        results[label] = {
            "strategy"  : strategy,
            "use_rag"   : use_rag,
            "n_runs"    : len(latencies),
            "mean_s"    : mean_lat,
            "p95_s"     : p95_lat,
            "min_s"     : min_lat,
            "max_s"     : max_lat,
            "raw"       : latencies,
        }
        print(f"    mean={mean_lat}s  p95={p95_lat}s\n")

    # Compute RAG overhead
    base_mean = results.get("Zero-Shot", {}).get("mean_s", 0)
    rag_mean  = results.get("Zero-Shot + RAG", {}).get("mean_s", 0)
    rag_overhead = round(rag_mean - base_mean, 3) if base_mean and rag_mean else 0

    # Print summary table
    print(f"{'='*60}")
    print(f"  BENCHMARK RESULTS  ({MODEL})")
    print(f"{'='*60}")
    print(f"  {'Condition':<22} {'Mean':>7} {'P95':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for label, r in results.items():
        print(f"  {label:<22} {r['mean_s']:>6.3f}s {r['p95_s']:>6.3f}s "
              f"{r['min_s']:>6.3f}s {r['max_s']:>6.3f}s")
    print(f"{'='*60}")
    print(f"  RAG overhead: +{rag_overhead:.3f}s vs zero_shot baseline")
    print(f"{'='*60}\n")

    # Save JSON
    output = {
        "run_date"    : datetime.now().isoformat(),
        "model"       : MODEL,
        "n_runs"      : N_RUNS,
        "conditions"  : results,
        "rag_overhead": rag_overhead,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results saved -> {OUTPUT_PATH.resolve()}")

    # Generate chart
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        labels    = list(results.keys())
        means     = [results[l]["mean_s"] for l in labels]
        p95s      = [results[l]["p95_s"]  for l in labels]
        x         = np.arange(len(labels))
        width     = 0.35
        colors    = ["#4C9BE8", "#4C9BE8", "#4C9BE8", "#E8834C"]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars1 = ax.bar(x - width/2, means, width, label="Mean",
                       color=colors, alpha=0.85, edgecolor="white")
        bars2 = ax.bar(x + width/2, p95s,  width, label="P95",
                       color=colors, alpha=0.5,  edgecolor="white", hatch="//")

        ax.set_xlabel("Condition")
        ax.set_ylabel("Latency (seconds)")
        ax.set_title(f"Latency Benchmark — {MODEL}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.legend()
        ax.set_ylim(0, max(p95s) * 1.3 if p95s else 5)
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with values
        for bar in bars1:
            ax.annotate(f"{bar.get_height():.2f}s",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        rag_patch = mpatches.Patch(color="#E8834C", alpha=0.85,
                                   label=f"RAG overhead: +{rag_overhead:.3f}s")
        ax.legend(handles=ax.get_legend_handles_labels()[0] + [rag_patch])

        plt.tight_layout()
        plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Chart saved   -> {CHART_PATH.resolve()}")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    print("\nTask 3 complete!")


if __name__ == "__main__":
    main()