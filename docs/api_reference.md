# API Reference — AI SQL Query Assistant

**Base URL:** `http://localhost:5000`  
**Model:** `llama-3.3-70b-versatile` via Groq  
**Version:** Week 3  
**Auth:** None required (local dev server)

---

## Endpoints Overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status check |
| POST | `/generate` | Generate SQL from natural language |
| POST | `/evaluate` | Score generated SQL against expected SQL |

---

## GET /health

Check if the server is running and get model info.

**Request:**
```bash
curl http://localhost:5000/health
```

**PowerShell:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

**Response `200 OK`:**
```json
{
  "status": "ok",
  "model": "llama-3.3-70b-versatile",
  "strategies": ["zero_shot", "few_shot", "chain_of_thought"],
  "timestamp": "2026-04-04T16:33:42.372819"
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when server is healthy |
| `model` | string | Active LLM model name |
| `strategies` | array | Available prompt strategies |
| `timestamp` | string | ISO 8601 server time |

---

## POST /generate

Generate a SQL query from a natural language question.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `nl_query` | string | Yes | — | Natural language question |
| `schema` | object | No | `{}` | Database schema hint |
| `schema.tables` | array | No | `[]` | Table names |
| `schema.columns` | array | No | `[]` | Column names |
| `strategy` | string | No | `zero_shot` | Prompt strategy |

**Valid strategies:** `zero_shot` `few_shot` `chain_of_thought`

**Request — minimal:**
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d "{\"nl_query\": \"How many airports do we have?\"}"
```

**PowerShell — minimal:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"nl_query": "How many airports do we have?"}'
```

**PowerShell — with schema and strategy:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/generate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "nl_query": "Count airports by country ordered from most to least",
    "strategy": "few_shot",
    "schema": {
      "tables": ["airports"],
      "columns": ["airport_id", "city", "country", "airport_name"]
    }
  }'
```

**Response `200 OK`:**
```json
{
  "sql": "SELECT country, COUNT(*) FROM airports GROUP BY country ORDER BY COUNT(*) DESC",
  "raw": "SELECT country, COUNT(*) FROM airports GROUP BY country ORDER BY COUNT(*) DESC",
  "valid": true,
  "strategy": "zero_shot",
  "model": "llama-3.3-70b-versatile",
  "latency_s": 1.557,
  "parse_issues": []
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `sql` | string | Cleaned, normalized SQL query |
| `raw` | string | Raw LLM output before parsing |
| `valid` | bool | Whether `sql` is syntactically valid |
| `strategy` | string | Prompt strategy used |
| `model` | string | LLM model used |
| `latency_s` | float | API call duration in seconds |
| `parse_issues` | array | List of fixes applied by OutputParser |

**Error responses:**

| Status | Cause | Example |
|---|---|---|
| `400` | Missing `nl_query` | `{"error": "nl_query is required..."}` |
| `400` | Empty `nl_query` | `{"error": "nl_query is required..."}` |
| `400` | Invalid `strategy` | `{"error": "Invalid strategy 'x'..."}` |
| `500` | LLM API failure | `{"error": "LLM API call failed: ..."}` |
| `500` | Missing API key | `{"error": "GROQ_API_KEY not set..."}` |

---

## POST /evaluate

Score a generated SQL against an expected SQL.
Optionally generates the SQL first if `generated_sql` is not provided.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `nl_query` | string | Yes | — | Natural language question |
| `expected_sql` | string | Yes | — | Ground truth SQL |
| `generated_sql` | string | No | — | Pre-generated SQL to score (skips LLM call) |
| `schema` | object | No | `{}` | Database schema hint |
| `strategy` | string | No | `zero_shot` | Strategy used if generating SQL |

**PowerShell — generate then evaluate:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/evaluate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "nl_query": "How many airports do we have?",
    "expected_sql": "SELECT COUNT(*) FROM airports",
    "schema": {"tables": ["airports"], "columns": ["airport_id", "city", "country"]}
  }'
```

**PowerShell — score pre-generated SQL (no LLM call):**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/evaluate `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "nl_query": "How many airports do we have?",
    "expected_sql": "SELECT COUNT(*) FROM airports",
    "generated_sql": "SELECT COUNT(*) FROM airports"
  }'
```

**Response `200 OK`:**
```json
{
  "nl_query": "How many airports do we have?",
  "expected_sql": "SELECT COUNT(*) FROM airports",
  "generated_sql": "SELECT COUNT(*) FROM airports",
  "verdict": "PASS",
  "exact_match": true,
  "valid_sql": true,
  "score_notes": ["exact_structural_match"],
  "latency_s": 0.34,
  "strategy": "zero_shot"
}
```

**Response fields:**

| Field | Type | Description |
|---|---|---|
| `nl_query` | string | Original question |
| `expected_sql` | string | Ground truth SQL |
| `generated_sql` | string | Generated or provided SQL |
| `verdict` | string | `PASS` / `PARTIAL` / `FAIL` |
| `exact_match` | bool | True if SQLs are structurally identical |
| `valid_sql` | bool | True if generated SQL is syntactically valid |
| `score_notes` | array | Scoring observations |
| `latency_s` | float | LLM call duration (`null` if pre-generated) |
| `strategy` | string | Strategy used |

**Verdict definitions:**

| Verdict | Meaning |
|---|---|
| `PASS` | Generated SQL is structurally identical to expected SQL |
| `PARTIAL` | Generated SQL is valid but differs from expected |
| `FAIL` | Generated SQL is syntactically invalid or empty |

**Error responses:**

| Status | Cause |
|---|---|
| `400` | Missing `nl_query` |
| `400` | Missing `expected_sql` |
| `500` | LLM API failure |

---

## Scoring Details

The `/evaluate` endpoint uses a 2-metric scoring system:

**Metric 1 — Exact Match (via sqlglot):**
Both SQLs are normalized through `OutputParser` then parsed by sqlglot into an AST.
Identifiers are lowercased before comparison so `AIRPORTS` and `airports` match.
Quote style (`'value'` vs `"value"`) is normalized on both sides.

**Metric 2 — Valid SQL:**
The generated SQL is parsed by sqlglot. If it raises a `ParseError`, the SQL
is invalid and the verdict is `FAIL`.

---

## CLI Quick Reference

```bash
# Health check
python src/cli.py --health

# Single query
python src/cli.py --query "How many airports do we have?"

# With strategy
python src/cli.py --query "Count airports by country" --strategy few_shot

# With schema
python src/cli.py --query "Count airports" --tables airports --columns airport_id city country

# Evaluate a pair
python src/cli.py --evaluate `
  --query "How many airports?" `
  --expected "SELECT COUNT(*) FROM airports"

# Interactive mode
python src/cli.py
```

---

## Test Suite

```bash
# Unit tests (parser only, no server needed)
pytest tests/test_parser.py -v       # 33 tests

# API integration tests (server must be running)
python src/api.py                    # terminal 1
pytest tests/test_api.py -v         # terminal 2 -- 21 tests
```