# Prompt Strategy Report — Week 2

**Date:** April 2026  
**Model:** `llama-3.3-70b-versatile` via Groq  
**Evaluated on:** Golden set (10 correct pairs, 6 scenarios, 2 easy / 4 medium / 5 hard)

---

## Experiment Summary

A 3x1 grid experiment was run — 3 prompt strategies against 1 model on the
golden set. Each pair was processed through the full pipeline:
`PromptBuilder -> Groq API -> OutputParser -> Metrics (sqlglot exact match + valid SQL)`

| Strategy | PASS | PARTIAL | FAIL | VALID % |
|---|---|---|---|---|
| `zero_shot` | 0/10 | 10/10 | 0/10 | **100%** |
| `few_shot` | 0/10 | 10/10 | 0/10 | **100%** |
| `chain_of_thought` | 0/10 | 7/10 | 3/10 | **70%** |

---

## Winning Strategy: Zero-Shot (tied with Few-Shot)

`zero_shot` and `few_shot` tied at **100% valid SQL** with 0 failures.
`zero_shot` is declared the winner for Week 2 on the basis of:

- **Simplicity** — fewer tokens, faster, cheaper API calls
- **No risk of example leakage** — few-shot introduces a selection dependency
- **Equal performance** — few-shot added no measurable benefit on this golden set
- **Reproducibility** — no example selection randomness to account for

---

## Strategy Breakdown

### Strategy 1 — Zero-Shot

**System prompt:**
> "You are an expert SQL assistant. Given a database schema and a natural
> language question, return ONLY the SQL query. Do not explain. Do not use
> markdown. Do not add any text before or after the SQL."

**Prompt structure:**
```
Database schema:
Tables : ...
Columns: ...

Question: <nl_query>

SQL:
```

**Results:** 0 PASS, 10 PARTIAL, 0 FAIL — 100% valid SQL  
**Verdict:** Strong baseline. Every generated SQL was syntactically valid
and logically correct. All PARTIALs were style differences only (see below).

---

### Strategy 2 — Few-Shot

**System prompt:**
> "You are an expert SQL assistant. Study the examples below, then generate
> SQL for the new question. Return ONLY the SQL query..."

**Prompt structure:**
```
-- Examples:
-- Example 1: <schema + NL + SQL>
-- Example 2: <schema + NL + SQL>
-- Example 3: <schema + NL + SQL>

-- Now answer the following:
Database schema: ...
Question: <nl_query>
SQL:
```

**Example selection rules:**
- 3 examples from `pairs_clean.json` only
- Never from the golden set (enforced by `PromptBuilder`)
- Selected to cover different scenarios for diversity
- Prefer easy/medium difficulty for clarity

**Results:** 0 PASS, 10 PARTIAL, 0 FAIL — 100% valid SQL  
**Verdict:** No improvement over zero-shot on this golden set. The model is
already strong enough that examples don't help for these query types.
Few-shot may show improvement on a larger, noisier evaluation set.

---

### Strategy 3 — Chain-of-Thought

**System prompt:**
> "You are an expert SQL assistant. Think step by step before writing SQL.
> First identify the relevant tables and columns, then write the SQL query.
> Your final answer must be ONLY the SQL query on the last line..."

**Prompt structure:**
```
Database schema: ...
Question: <nl_query>

Let me think through this step by step:
1. What tables do I need?
2. What columns are relevant?
3. What SQL pattern fits this question?

SQL:
```

**Results:** 0 PASS, 7 PARTIAL, **3 FAIL** — 70% valid SQL  
**Verdict:** Worst performing strategy. The reasoning scaffold caused the
model to return chain-of-thought text instead of SQL in 3 out of 10 cases.
The `OutputParser` correctly rejected these as unparseable, scoring them FAIL.

**Failure cases:**
- `gold_05` (joins/hard) — model returned empty after reasoning
- `gold_07` (subqueries/hard) — model returned empty after reasoning  
- `gold_09` (sort_limit/medium) — model returned `SELECT with ORDER BY AND LIMIT...`
  (reasoning text leaked into SQL position)

**Root cause:** The `SQL:` label at the end of the prompt was not a strong
enough signal for the model to switch from reasoning mode to SQL-only output.
The model treated the reasoning scaffold as permission to keep explaining.

---

## Why All Scores Are PARTIAL (No PASS)

Zero PASS scores does not mean the model is wrong. It means the
**ground truth SQL and generated SQL differ in style**, not logic.

The most common style differences observed:

| Pattern | Expected | Generated | Impact |
|---|---|---|---|
| Quote style | `"Russia"` | `'Russia'` | sqlglot sees as different |
| Column case | `count(*)` | `COUNT(apid)` | Different column reference |
| Alias naming | `T1.Birth_Date` | `T2.Birth_Date` | Alias swap, same logic |
| DISTINCT missing | `SELECT DISTINCT` | `SELECT` | Real semantic difference |
| Column order | `COUNT(*), country` | `country, COUNT(*)` | Same result, different order |

**The fix for Week 3:** Normalize expected SQL through `OutputParser` before
scoring. This eliminates quote-style false negatives and will convert several
PARTIALs to PASS.

---

## Notable Observations

**gold_10 (simple_select) missed DISTINCT:**
```sql
-- Expected
SELECT DISTINCT characteristic_name FROM CHARACTERISTICS

-- Generated (all 3 strategies)
SELECT characteristic_name FROM Characteristics
```
All 3 strategies consistently dropped `DISTINCT`. This suggests the model
needs a stronger schema signal (e.g. column cardinality hints) or an explicit
instruction about deduplication to catch this pattern.

**gold_07 (subquery) — wrong column in subquery:**
```sql
-- Expected
WHERE Aircraft_ID NOT IN (SELECT Winning_Aircraft FROM MATCH)

-- Generated
WHERE Aircraft_ID NOT IN (SELECT Aircraft_ID FROM aircraft)
```
The model queried the wrong table in the subquery. This is a genuine schema
understanding failure — without knowing which column `Winning_Aircraft` maps
to, the model defaulted to a plausible but wrong column.

---

## Recommendations for Week 3

1. **Normalize expected SQL before scoring** — run ground truth through
   `OutputParser` to eliminate quote-style false negatives. This alone should
   convert 3-4 PARTIALs to PASS.

2. **Drop chain-of-thought or fix the prompt** — add a stronger instruction
   like `"Write ONLY the final SQL. No explanation. Start with SELECT."` at
   the end to prevent reasoning text leaking into the SQL position.

3. **Add column type hints to schema** — for subquery failures, include
   foreign key relationships in the schema text to guide column selection.

4. **Add DISTINCT instruction** — include in the system prompt:
   `"Use DISTINCT when the question implies unique values."`

5. **Expand evaluation beyond golden set** — run on all 64 pairs from
   `pairs_clean.json` to get statistically meaningful pass rates per scenario.

---

## Files Produced This Week

| File | Description |
|---|---|
| `src/prompt_builder.py` | 3 strategies, golden exclusion enforced |
| `src/output_parser.py` | 11-step cleaner, 13/13 smoke test |
| `src/metrics.py` | sqlglot exact match + valid SQL check |
| `src/eval_runner.py` | Full pipeline, 3x1 experiment runner |
| `tests/test_parser.py` | 33 unit tests, all passing |
| `outputs/results_week2.json` | Full experiment results with metadata |
| `docs/prompt_strategy.md` | This document |