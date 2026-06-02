# AI SQL Assistant

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![HF Spaces](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Krish8854/ai-sql-assistant)
[![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen)]()

> Natural language to SQL using RAG + LLaMA-3.3-70B.
> **+55pp exact match improvement** with retrieval augmentation.

## Live Demo

**[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Krish8854/ai-sql-assistant)**

---

## Architecture

```
NL Query
    |
    v
[Retriever] ── top-3 similar pairs ──> [PromptBuilder]
    |                                        |
sentence-transformers                   strategy +
cosine similarity                       few-shot examples
over 64 embeddings                           |
                                             v
                                   [LLaMA-3.3-70B via Groq]
                                             |
                                             v
                                      [OutputParser]
                                      11-step cleaner
                                             |
                                             v
                                       Valid SQL + Score
```

---

## Quickstart

```bash
git clone https://github.com/Krish8854/ai-sql-assistant
cd ai-sql-assistant
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY (free at console.groq.com)

python src/embed_pairs.py     # build embeddings index (first time only)
python src/api.py             # Terminal 1 -- Flask API
streamlit run src/app.py      # Terminal 2 -- Streamlit UI
```

---

## Results

| Condition | PASS% | Semantic Sim | Valid SQL% |
|---|---|---|---|
| Baseline (no RAG) | 12% | 0.9287 | 89% |
| RAG (k=3) | **67%** | **0.9869** | 91% |
| **Delta** | **+55pp** | **+0.058** | **+2pp** |

### Model Comparison (zero_shot + RAG, golden set)

| Model | PASS% | Avg Latency | Avg Semantic |
|---|---|---|---|
| LLaMA-3.3-70B | 20% | 0.25s | 0.9597 |
| LLaMA-3.1-8B | 20% | 0.23s | 0.9277 |

---

## Demo

**Example 1 — Aggregation with GROUP BY:**
```
NL:  What is the total revenue per product category, ordered highest to lowest?
SQL: SELECT category, SUM(price * quantity) AS total_revenue
     FROM orders JOIN products ON orders.product_id = products.id
     GROUP BY category ORDER BY total_revenue DESC
```

**Example 2 — Subquery with NOT IN:**
```
NL:  Find all customers who have never placed an order.
SQL: SELECT id, name FROM customers
     WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)
```

**Example 3 — Top-N with JOIN:**
```
NL:  Who are the top 5 customers by total spend?
SQL: SELECT c.name, SUM(o.amount) AS total_spend
     FROM customers c JOIN orders o ON c.id = o.customer_id
     GROUP BY c.name ORDER BY total_spend DESC LIMIT 5
```

---

## Demo Video

[![Demo Video](https://img.shields.io/badge/▶-Watch%20Demo-red)](https://youtu.be/pFNU2dfEEls?si=Hhh8e4kMV8qdpgG8)

**[Watch the 3-min demo on YouTube →](https://youtu.be/pFNU2dfEEls?si=Hhh8e4kMV8qdpgG8)**

---

## How It Works

### 1. Dataset Pipeline
- Loads Spider (multi-table) and WikiSQL (single-table) benchmarks
- 64 balanced NL->SQL pairs across 6 scenario types
- 13-check automated quality validator
- Golden set of 10 hand-labeled pairs for evaluation

### 2. RAG Pipeline
1. NL query encoded using `all-MiniLM-L6-v2` (384-dim)
2. Cosine similarity against 64 pre-indexed pair embeddings
3. Top-3 most similar pairs retrieved (`exclude_ids` prevents self-retrieval)
4. Retrieved pairs injected as few-shot examples into prompt

### 3. Output Parser
11-step cleaning pipeline: markdown fences, explanation text, lowercase
keywords, trailing semicolons, double quotes, numbered prefixes,
multiple statements, whitespace normalization.

### 4. Scoring Metrics
- **Exact match** -- sqlglot AST normalization + identifier lowercasing
- **Semantic similarity** -- sentence-transformers cosine similarity
- **Valid SQL** -- sqlglot parse validation

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status |
| GET | `/metrics/summary` | Aggregate eval stats |
| POST | `/generate` | NL -> SQL (supports `use_rag: true`) |
| POST | `/evaluate` | Score SQL against expected |

```powershell
# Generate with RAG
Invoke-RestMethod -Uri http://localhost:5000/generate `
  -Method POST -ContentType "application/json" `
  -Body '{"nl_query": "Count airports by country", "use_rag": true}'
```

---

## Evaluation

```bash
# Run 64-pair eval with RAG
python src/eval_runner.py --strategy zero_shot --full-pairs --rag

# Latency benchmark
python src/benchmark.py

# Model comparison
python src/model_comparison.py

# All 82 tests
pytest tests/ -v
```

---

## Project Structure

```
ai-sql-assistant/
|-- src/
|   |-- api.py            # Flask REST API (4 endpoints)
|   |-- app.py            # Streamlit UI (Generate/Evaluate/History)
|   |-- retriever.py      # Cosine similarity RAG retriever
|   |-- prompt_builder.py # 3 prompt strategies
|   |-- output_parser.py  # 11-step LLM output cleaner
|   |-- metrics.py        # Exact match + semantic similarity
|   |-- eval_runner.py    # Eval pipeline + RAG comparison
|   |-- embed_pairs.py    # Generate embeddings index
|   |-- model_comparison.py # Multi-model benchmark
|   |-- benchmark.py      # Latency benchmark
|   |-- loader.py         # Spider + WikiSQL data loader
|   |-- collect.py        # Sample balanced pairs
|   |-- clean.py          # Normalize + validate pairs
|   +-- golden.py         # Build golden evaluation set
|
|-- data/
|   |-- processed/        # pairs_clean.json (64 pairs)
|   |-- golden/           # golden.json (10 correct + 1 wrong)
|   +-- embeddings/       # pairs_embeddings.npy (64x384)
|
|-- outputs/
|   |-- results_week3.json         # Baseline eval
|   |-- results_week4.json         # RAG vs baseline (golden)
|   |-- results_week5_rag_full.json # RAG vs baseline (64 pairs)
|   |-- model_comparison.json       # 70B vs 8B
|   +-- benchmark_results.json      # Latency benchmark
|
|-- tests/
|   |-- test_parser.py    # 33 OutputParser unit tests
|   |-- test_api.py       # 21 Flask API integration tests
|   +-- test_app_smoke.py # 28 Streamlit component checks
|
|-- docs/
|   |-- evaluation_report.md  # Full 6-week eval report
|   |-- case_study.md         # Project narrative
|   |-- api_reference.md      # API documentation
|   |-- prompt_strategy.md    # Prompt experiment results
|   +-- interview_prep.md     # 5 Q&A pairs
|
|-- .env.example
|-- requirements.txt
+-- README.md
```

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM inference | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| SQL normalization | sqlglot |
| API framework | Flask + flask-cors |
| UI | Streamlit |
| Testing | pytest (82 tests) |
| Deployment | Hugging Face Spaces |
| Datasets | Spider + WikiSQL |

---

## Weekly Sprint Log

| Week | Focus | Key Result | Score |
|---|---|---|---|
| 1 | Data pipeline | 64 pairs, golden set, 13-check validator | 100/100 |
| 2 | Prompt engineering | 3 strategies, OutputParser, 33 tests | 100/100 |
| 3 | Flask API + CLI | 4 endpoints, 21 integration tests, 91% valid SQL | 100/100 |
| 4 | RAG integration | Retriever, +50pp on golden set | 85/100 |
| 5 | Streamlit UI | 3-tab UI, 67% PASS on 64 pairs | 85/100 |
| 6 | Evaluation + deployment | Full report, HF Spaces live | 80/100 |
| 7 | Portfolio packaging | GitHub push, benchmarks, case study | TBD |

---

## License

MIT License — see [LICENSE](LICENSE) for details.