# Case Study: AI SQL Code Assistant

**Author:** Mandal
**Duration:** 6 weeks (April 2026)
**Live Demo:** [huggingface.co/spaces/Krish8854/ai-sql-assistant](https://huggingface.co/spaces/Krish8854/ai-sql-assistant)

---

## Problem

Non-technical users struggle to write SQL queries even when they know
exactly what data they need. Existing tools require SQL knowledge as a
prerequisite, creating a bottleneck in data workflows where analysts
and business users depend on engineers for basic data retrieval.

---

## Approach

Built a retrieval-augmented NL->SQL system that:

1. Accepts a natural language query and optional schema hint
2. Encodes the query using sentence-transformers (all-MiniLM-L6-v2)
3. Retrieves the top-3 most similar NL->SQL pairs via cosine similarity
4. Injects retrieved pairs as few-shot examples into the LLM prompt
5. Cleans the raw LLM output through an 11-step OutputParser
6. Returns syntactically valid, normalized SQL with a confidence verdict

---

## Technical Stack

| Component | Technology |
|---|---|
| LLM | LLaMA-3.3-70B via Groq API (free tier) |
| Retrieval | sentence-transformers all-MiniLM-L6-v2 |
| Similarity | Cosine similarity over 64 L2-normalized embeddings |
| SQL validation | sqlglot AST normalization + identifier lowercasing |
| API | Flask REST (3 endpoints + /metrics/summary) |
| UI | Streamlit (3 tabs: Generate, Evaluate, History) |
| Evaluation | Exact match + semantic similarity + valid SQL% |
| Deployment | Hugging Face Spaces (public, free) |
| Test coverage | 82 tests across 3 test suites (100% pass rate) |

---

## Results

| Condition | Exact Match | Semantic Similarity | Valid SQL% |
|---|---|---|---|
| Baseline (no RAG) | 12% | 0.9287 | 89% |
| RAG (k=3, golden set) | 60% | 0.9902 | 100% |
| RAG (k=3, 64 pairs) | **67%** | **0.9869** | 91% |
| **Improvement** | **+55pp** | **+0.058** | **+2pp** |

### Model Comparison (zero_shot + RAG, golden set)

| Model | PASS% | Valid SQL% | Avg Latency | Avg Semantic |
|---|---|---|---|---|
| LLaMA-3.3-70B | 20% | 100% | 0.25s | 0.9597 |
| LLaMA-3.1-8B | 20% | 100% | 0.23s | 0.9277 |

---

## Key Findings

**1. RAG is the dominant driver of improvement.**
Prompt strategy choice (zero_shot vs few_shot vs chain_of_thought)
had minimal impact on accuracy. Retrieval quality — getting the right
examples into the prompt — mattered far more than how the prompt was
structured. This suggests that for structured output tasks like SQL
generation, context is more valuable than reasoning scaffolds.

**2. Model size matters less than expected at this scale.**
LLaMA-3.1-8B matched LLaMA-3.3-70B on exact match rate (20% each).
The 70B model's advantage appears only in semantic similarity (+0.032),
suggesting it produces more idiomatic SQL. For latency-critical
applications, the 8B model is a viable alternative.

**3. Exact match understates quality significantly.**
~40% of non-PASS results are style variations (alias naming, column
ordering, COUNT(*) vs COUNT(col)) that produce semantically identical
SQL. Semantic similarity scores of 0.90+ on these cases confirm the
model is logically correct even when not an exact string match.

---

## Failure Analysis

| Mode | Count | Fixed? | Mitigation |
|---|---|---|---|
| DISTINCT dropped | ~8/64 | Partial | System prompt instruction added |
| CoT reasoning leak | 3/10 | Yes | "Start with SELECT" instruction |
| Quote style mismatch | ~6/64 | Yes | OutputParser normalization |
| Alias swap (T1/T2) | ~5/64 | No | Acceptable — semantically equivalent |
| COUNT(*) vs COUNT(col) | ~4/64 | No | Acceptable — functionally equivalent |

---

## What I'd Do Differently

- **Normalize ground truth SQL at collection time (Week 1)** — would have
  given reliable exact match metrics from the start instead of discovering
  the issue in Week 4.
- **Build a minimal UI in Week 2** — makes results tangible sooner and
  helps identify UX issues earlier in the development cycle.
- **Use semantic similarity as the primary metric from day one** — exact
  match is too brittle for SQL comparison. Semantic similarity better
  reflects real-world correctness.
- **Expand the dataset to 500+ pairs** — retrieval quality is bounded by
  index coverage. 64 pairs is sufficient for a prototype but limits RAG
  effectiveness on edge cases.

---

## Live Demo

**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Krish8854/ai-sql-assistant)**

Enter any natural language question about a database. Toggle RAG on to
see retrieved examples alongside the generated SQL.