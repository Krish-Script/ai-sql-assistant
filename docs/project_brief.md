# AI SQL Query Assistant

A domain-specific AI assistant that converts natural language questions into
valid SQL queries, with RAG (Retrieval-Augmented Generation) for improved accuracy.

Built over 4 weekly sprints using Python, Flask, sentence-transformers, and LLaMA-3.3
via Groq. Evaluated on Spider and WikiSQL benchmark datasets.

---

## Architecture
```
User NL Query
      |
      v
 PromptBuilder  <-- RAG Retriever (top-3 similar pairs)
      |                        |
      |                sentence-transformers
      |                cosine similarity
      |                over 64 embeddings
      v
  Groq API(llama-3.3-70b-versatile)
      |
      v
 OutputParser  (strips fences, normalizes keywords, validates)
      |
      v
   Metrics  (exact match via sqlglot + semantic similarity)
      |
      v
  Flask API  /generate  /evaluate  /health
      |
      v
   CLI Client  (interactive + single query modes)
```

---

## Quickstart


# 1. Install dependencies
pip install flask flask-cors groq sentence-transformers sqlglot scikit-learn numpy python-dotenv requests pytest

# 2. Add your Groq API key (free at console.groq.com)
echo "GROQ_API_KEY=your-key-here" > .env

# 3. Build embeddings index (first time only)
python src/embed_pairs.py

# 4. Start the API
python src/api.py


### Then in a second terminal
# Generate SQL from natural language
python src/cli.py --query "How many airports do we have?"

# With RAG enabled
python src/cli.py --query "Count airports by country" --strategy few_shot

# Interactive mode
python src/cli.py


---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status + model info |
| POST | `/generate` | NL -> SQL generation |
| POST | `/evaluate` | Score generated SQL vs expected |

**Generate SQL (with RAG):**

Invoke-RestMethod -Uri http://localhost:5000/generate `
  -Method POST -ContentType "application/json" `
  -Body '{"nl_query": "Count airports by country", "use_rag": true}'


**Evaluate a pair:**

Invoke-RestMethod -Uri http://localhost:5000/evaluate `
  -Method POST -ContentType "application/json" `
  -Body '{
    "nl_query": "How many airports?",
    "expected_sql": "SELECT COUNT(*) FROM airports",
    "generated_sql": "SELECT COUNT(*) FROM airports"
  }'


---

## How It Works

### 1. Dataset Pipeline
- Loads **Spider** (multi-table JSON) and **WikiSQL** (single-table Parquet)
- Samples 64 balanced NL->SQL pairs across 6 scenario types
- Normalizes SQL keywords, flattens schema, derives difficulty
- Validates with 13 automated quality checks

### 2. Prompt Strategies
Three strategies implemented in `PromptBuilder`:
- **zero_shot** -- schema + question, no examples (fastest)
- **few_shot** -- 3 hand-picked examples + question
- **chain_of_thought** -- step-by-step reasoning scaffold

### 3. RAG Pipeline
When `use_rag: true` is passed to `/generate`:
1. The NL query is encoded using `all-MiniLM-L6-v2` (384-dim embeddings)
2. Cosine similarity is computed against 64 pre-indexed pair embeddings
3. Top-3 most similar NL->SQL pairs are retrieved
4. Retrieved pairs are injected as few-shot examples into the prompt
5. `exclude_ids` prevents the query's own pair from being retrieved

### 4. Output Parser
11-step cleaning pipeline handles all real LLM output patterns:
markdown fences, explanation text, lowercase keywords, trailing semicolons,
double quotes, numbered prefixes, multiple statements, and more.

### 5. Scoring Metrics
- **Exact match** -- sqlglot AST normalization + identifier lowercasing
- **Valid SQL** -- sqlglot parse validation (no live DB needed)
- **Semantic similarity** -- sentence-transformers cosine similarity between SQL strings

---

## Evaluation Results

### RAG vs Baseline (Golden Set, 10 pairs)

| Condition | PASS | VALID | Avg Semantic Sim |
|---|---|---|---|
| Baseline (zero_shot) | 1/10 (10%) | 100% | 0.9500 |
| RAG | **6/10 (60%)** | 100% | **0.9902** |
| Delta | **+5 PASS (+50pp)** | 0% | **+0.0402** |

### 64-pair Evaluation (zero_shot, fixed harness)

| Metric | Value |
|---|---|
| PASS (exact match) | 8/64 (12%) |
| PARTIAL (valid, not exact) | 50/64 (78%) |
| FAIL (invalid SQL) | 6/64 (9%) |
| Valid SQL | 58/64 (91%) |

### Key Finding
RAG improves exact match by 50pp on the golden set (10% to 60%).
Semantic similarity of 0.99 with RAG confirms the model generates
semantically correct SQL even when not an exact string match.

---

## Project Structure

```
ai-code-assistant/
|-- data/
|   |-- raw/              # Spider + WikiSQL source files
|   |-- processed/        # pairs_clean.json (64 pairs), pairs_clean.csv
|   |-- golden/           # golden.json (10 correct + 1 wrong pair)
|   +-- embeddings/       # pairs_embeddings.npy (64x384), pairs_index.json
|
|-- src/
|   |-- loader.py         # Loads Spider + WikiSQL into unified DataFrame
|   |-- collect.py        # Samples balanced NL->SQL pairs by scenario
|   |-- clean.py          # Normalizes SQL, schema, difficulty
|   |-- validator.py      # 13-check quality validator
|   |-- golden.py         # Builds golden evaluation set
|   |-- embed_pairs.py    # Generates sentence embeddings for all pairs
|   |-- retriever.py      # Cosine similarity RAG retriever
|   |-- prompt_builder.py # 3 prompt strategies
|   |-- output_parser.py  # 11-step LLM output cleaner
|   |-- metrics.py        # Exact match + semantic similarity scoring
|   |-- eval_runner.py    # Full eval pipeline + RAG comparison runner
|   |-- api.py            # Flask REST API
|   +-- cli.py            # CLI client
|
|-- tests/
|   |-- test_parser.py    # 33 unit tests for OutputParser
|   +-- test_api.py       # 21 integration tests for Flask API
|
|-- outputs/
|   |-- results_week3.json  # 64-pair baseline eval
|   +-- results_week4.json  # RAG vs baseline comparison
|
|-- docs/
|   |-- project_brief.md    # Scope, goals, constraints
|   |-- prompt_strategy.md  # Prompt experiment results
|   +-- api_reference.md    # Full API documentation
|
|-- .env.example
|-- requirements.txt
+-- README.md
```

---

## Test Suite


# Unit tests (no server needed) -- 33 tests
pytest tests/test_parser.py -v

# API integration tests (start server first) -- 21 tests
python src/api.py            # terminal 1
pytest tests/test_api.py -v  # terminal 2


---

## Dataset

| Source | Pairs | Scenarios |
|---|---|---|
| Spider | 47 | All 6 |
| WikiSQL | 17 | filters, aggregations, sort_limit, simple_select |
| **Total** | **64** | **6** |

**Scenarios:** filters, aggregations, joins, subqueries, sort_limit, simple_select

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM inference | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| SQL normalization | sqlglot |
| API framework | Flask + flask-cors |
| Testing | pytest |
| Datasets | Spider + WikiSQL |

---

## Weekly Sprint Log

| Week | Focus | Key Deliverable |
|---|---|---|
| 1 | Data pipeline | 64 clean NL->SQL pairs, golden set, validator |
| 2 | Prompt engineering | 3 strategies, OutputParser, 33 unit tests |
| 3 | API + CLI | Flask API, CLI client, 21 passing integration tests |
| 4 | RAG | Retriever, RAG endpoint, +50pp PASS improvement |

---

## Demo

The assistant handles a wide range of SQL patterns out of the box.
With RAG enabled, it retrieves semantically similar examples from the
evaluation dataset to guide generation — improving exact match accuracy
by +55 percentage points over the zero-shot baseline.

**Example 1 — Aggregation with GROUP BY:**

NL:  What is the total revenue per product category, ordered from highest to lowest?
SQL: SELECT category, SUM(price * quantity) AS total_revenue
     FROM orders JOIN products ON orders.product_id = products.id
     GROUP BY category ORDER BY total_revenue DESC

*Non-trivial: requires a JOIN across two tables + aggregate + sort.*

**Example 2 — Subquery with NOT IN:**

NL:  Find all customers who have never placed an order.
SQL: SELECT id, name FROM customers
     WHERE id NOT IN (SELECT DISTINCT customer_id FROM orders)

*Non-trivial: uses a correlated NOT IN subquery — a common LLM failure mode.*

**Example 3 — Window-style top-N per group:**

NL:  Who are the top 5 customers by total spend?
SQL: SELECT c.name, SUM(o.amount) AS total_spend
     FROM customers c JOIN orders o ON c.id = o.customer_id
     GROUP BY c.name ORDER BY total_spend DESC LIMIT 5

*Non-trivial: JOIN + aggregate + ORDER BY + LIMIT pattern.*

### RAG vs No-RAG Results

| Condition | PASS% | Semantic Sim | Valid SQL% |
|---|---|---|---|
| Baseline (no RAG) | 12% | 0.929 | 89% |
| RAG enabled (k=3) | **67%** | **0.987** | 91% |

> RAG delivers a **+55pp improvement** in exact match accuracy by retrieving
> the 3 most semantically similar NL->SQL pairs and injecting them as
> few-shot examples into the prompt.