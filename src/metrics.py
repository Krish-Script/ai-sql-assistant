"""
metrics.py -- Task 4: Scoring metrics for NL->SQL evaluation
Used by: eval_runner.py

Two metrics:
    1. exact_match   : structural SQL comparison via sqlglot AST normalization
                       fixes Week 1 PARTIAL problem (quote style, whitespace,
                       alias names causing false negatives)

    2. valid_sql     : checks if generated SQL is syntactically valid
                       using sqlglot parsing (safe alternative to DuckDB
                       execution for multi-table Spider schemas)

Verdict system:
    PASS    : exact_match = True
    PARTIAL : valid_sql = True but exact_match = False
    FAIL    : valid_sql = False (unparseable / empty)

Usage:
    from metrics import score
    result = score(generated="SELECT name FROM users", expected="SELECT name FROM users")
    print(result.verdict)   # PASS
    print(result.exact)     # True
"""

import re
import sys
from pathlib import Path
import sqlglot
import sqlglot.errors
from dataclasses import dataclass, field

# OutputParser used to normalize both sides before comparison
# Lazy import to avoid circular dependency issues
def _get_parser():
    sys.path.insert(0, str(Path(__file__).parent))
    from output_parser import OutputParser
    return OutputParser()


# -------------------------------------------------------
# ScoreResult
# -------------------------------------------------------

@dataclass
class ScoreResult:
    """
    Result of scoring a generated SQL against an expected SQL.

    Attributes:
        generated       : cleaned generated SQL string
        expected        : expected SQL string
        exact           : True if AST-normalized SQLs match
        valid           : True if generated SQL is syntactically valid
        verdict         : PASS | PARTIAL | FAIL
        normalized_gen  : sqlglot-normalized form of generated SQL
        normalized_exp  : sqlglot-normalized form of expected SQL
        parse_error     : error message if generated SQL failed to parse
        notes           : list of observations about the comparison
    """
    generated     : str
    expected      : str
    exact         : bool
    valid         : bool
    verdict       : str
    normalized_gen: str        = ""
    normalized_exp: str        = ""
    parse_error   : str        = ""
    notes         : list[str]  = field(default_factory=list)

    def __str__(self) -> str:
        icon = "+" if self.verdict == "PASS" else "~" if self.verdict == "PARTIAL" else "x"
        return (
            f"[{icon}] {self.verdict}  "
            f"exact={self.exact}  valid={self.valid}"
            + (f"  error={self.parse_error}" if self.parse_error else "")
        )


# -------------------------------------------------------
# Core normalizer
# -------------------------------------------------------

def _normalize_with_sqlglot(sql: str, dialect: str = "") -> tuple[str, str]:
    """
    Parse SQL with sqlglot and return a canonical string form.
    Returns (normalized_sql, error_message).
    Empty error_message means success.
    """
    if not sql or not sql.strip():
        return "", "empty input"

    try:
        # Parse then regenerate — this canonicalizes formatting,
        # keyword casing, quote style, and whitespace
        parsed = sqlglot.parse_one(sql, dialect=dialect or None)
        normalized = parsed.sql(dialect=dialect or None, pretty=False)
        # Lowercase identifiers for case-insensitive comparison
        normalized = _lowercase_identifiers(normalized)
        return normalized.strip(), ""
    except sqlglot.errors.ParseError as e:
        return "", f"ParseError: {str(e)[:120]}"
    except Exception as e:
        return "", f"Error: {str(e)[:120]}"


def _fallback_normalize(sql: str) -> str:
    """
    Lightweight fallback normalization when sqlglot can't parse.
    Used for partial comparison only — not for exact match.
    Uppercases keywords and collapses whitespace.
    """
    sql = re.sub(r'\s+', ' ', sql.strip().upper())
    return sql


def _lowercase_identifiers(sql: str) -> str:
    """
    Lowercase all unquoted identifiers while keeping SQL keywords uppercase.
    Handles table/column name case differences like:
    CHARACTERISTICS vs Characteristics vs characteristics
    """
    SQL_KEYWORDS = {
        "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS",
        "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
        "ON", "AS", "DISTINCT", "ORDER", "BY", "GROUP", "HAVING",
        "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT", "CASE",
        "WHEN", "THEN", "ELSE", "END", "NULL", "IS", "LIKE", "BETWEEN",
        "COUNT", "SUM", "AVG", "MAX", "MIN", "ASC", "DESC",
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
        "TABLE", "INTO", "VALUES", "SET", "PRIMARY", "KEY",
        "FOREIGN", "REFERENCES", "COALESCE", "OVER", "PARTITION",
    }

    def lower_if_not_keyword(match):
        token = match.group(0)
        if token.upper() in SQL_KEYWORDS:
            return token
        return token.lower()

    return re.sub(r'\b[A-Za-z_][A-Za-z0-9_]*\b', lower_if_not_keyword, sql)


# -------------------------------------------------------
# Metric 1: Exact Match
# -------------------------------------------------------

def exact_match(generated: str, expected: str) -> tuple[bool, str, str, str]:
    """
    Compare two SQL strings structurally using sqlglot AST normalization.
    Returns (is_match, normalized_generated, normalized_expected, error).

    Week 3 fix: both sides are run through OutputParser first,
    THEN through sqlglot. This ensures quote style ('Russia' vs "Russia"),
    keyword casing, whitespace and trailing semicolons are all normalized
    on BOTH sides before comparison — eliminating false PARTIAL scores.
    """
    # Week 3 fix: run both sides through OutputParser first
    # This normalizes quote style, keyword casing, whitespace on both sides
    parser = _get_parser()
    gen_parsed = parser.parse(generated)
    exp_parsed = parser.parse(expected)

    # Use parser output if successful, else use raw input
    gen_clean = gen_parsed.sql if gen_parsed.success else generated
    exp_clean = exp_parsed.sql if exp_parsed.success else expected

    norm_gen, err_gen = _normalize_with_sqlglot(gen_clean)
    norm_exp, err_exp = _normalize_with_sqlglot(exp_clean)

    if err_gen or err_exp:
        # Fall back to basic string comparison if sqlglot fails on either
        fb_gen = _fallback_normalize(gen_clean)
        fb_exp = _fallback_normalize(exp_clean)
        return fb_gen == fb_exp, fb_gen, fb_exp, err_gen or err_exp

    return norm_gen == norm_exp, norm_gen, norm_exp, ""


# -------------------------------------------------------
# Metric 2: Valid SQL Check
# -------------------------------------------------------

def valid_sql(sql: str) -> tuple[bool, str]:
    """
    Check if a SQL string is syntactically valid using sqlglot.
    Returns (is_valid, error_message).

    This is the safe alternative to DuckDB execution for Week 2.
    Spider schemas are not loaded into any DB so execution would fail
    even for correct SQL — sqlglot parse-validation avoids this.
    """
    if not sql or not sql.strip():
        return False, "empty SQL"

    _, error = _normalize_with_sqlglot(sql)
    if error:
        return False, error
    return True, ""


# -------------------------------------------------------
# Combined scorer
# -------------------------------------------------------

def score(generated: str, expected: str) -> ScoreResult:
    """
    Score a generated SQL against an expected SQL.
    Runs both metrics and returns a ScoreResult with verdict.

    Verdict logic:
        PASS    : exact_match = True  (structurally identical)
        PARTIAL : exact_match = False but valid_sql = True
        FAIL    : valid_sql = False (unparseable or empty)

    Args:
        generated : cleaned SQL from LLM (after OutputParser)
        expected  : ground truth SQL from dataset

    Returns:
        ScoreResult
    """
    notes = []

    # Handle empty generated SQL (parse failure upstream)
    if not generated or not generated.strip():
        return ScoreResult(
            generated    = generated,
            expected     = expected,
            exact        = False,
            valid        = False,
            verdict      = "FAIL",
            parse_error  = "empty generated SQL",
            notes        = ["OutputParser returned empty string"],
        )

    # Metric 2: Valid SQL
    is_valid, parse_error = valid_sql(generated)

    # Metric 1: Exact match
    is_exact, norm_gen, norm_exp, norm_error = exact_match(generated, expected)

    # Build notes
    if is_exact:
        notes.append("exact_structural_match")
    else:
        # Explain why it's not an exact match
        if norm_gen and norm_exp:
            gen_tokens = set(norm_gen.upper().split())
            exp_tokens = set(norm_exp.upper().split())
            missing = exp_tokens - gen_tokens
            extra   = gen_tokens - exp_tokens
            if missing:
                notes.append(f"missing_tokens: {sorted(missing)[:5]}")
            if extra:
                notes.append(f"extra_tokens: {sorted(extra)[:5]}")

    if norm_error:
        notes.append(f"normalization_fallback: {norm_error[:60]}")

    # Verdict
    if is_exact:
        verdict = "PASS"
    elif is_valid:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"

    return ScoreResult(
        generated     = generated,
        expected      = expected,
        exact         = is_exact,
        valid         = is_valid,
        verdict       = verdict,
        normalized_gen= norm_gen,
        normalized_exp= norm_exp,
        parse_error   = parse_error,
        notes         = notes,
    )


# -------------------------------------------------------
# Batch scorer
# -------------------------------------------------------

def score_all(pairs: list[tuple[str, str]]) -> list[ScoreResult]:
    """
    Score a list of (generated, expected) pairs.

    Args:
        pairs: list of (generated_sql, expected_sql) tuples

    Returns:
        list of ScoreResult
    """
    return [score(gen, exp) for gen, exp in pairs]


def summarize(results: list[ScoreResult]) -> dict:
    """
    Summarize a list of ScoreResults into aggregate stats.

    Returns:
        dict with pass/partial/fail counts and rates
    """
    total   = len(results)
    passed  = sum(1 for r in results if r.verdict == "PASS")
    partial = sum(1 for r in results if r.verdict == "PARTIAL")
    failed  = sum(1 for r in results if r.verdict == "FAIL")

    return {
        "total"       : total,
        "pass"        : passed,
        "partial"     : partial,
        "fail"        : failed,
        "pass_rate"   : round(passed  / total, 3) if total else 0,
        "partial_rate": round(partial / total, 3) if total else 0,
        "fail_rate"   : round(failed  / total, 3) if total else 0,
        "valid_rate"  : round((passed + partial) / total, 3) if total else 0,
    }


# -------------------------------------------------------
# Semantic similarity (Week 4)
# -------------------------------------------------------

_sem_model = None   # loaded lazily on first call

def _get_sem_model():
    """Lazy-load the sentence-transformers model for semantic similarity."""
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sem_model


def semantic_similarity(generated: str, expected: str) -> float:
    """
    Compute cosine similarity between generated and expected SQL strings
    using sentence-transformers embeddings.

    Returns a float in [0.0, 1.0].
    0.0 = completely different, 1.0 = identical meaning.

    This metric captures semantic correctness that exact match misses:
    - COUNT(*) vs COUNT(col) -> high similarity
    - Same logic, different alias names -> high similarity
    - Completely wrong table -> low similarity

    Args:
        generated : cleaned generated SQL string
        expected  : expected SQL string

    Returns:
        float cosine similarity score
    """
    if not generated or not expected:
        return 0.0

    try:
        model  = _get_sem_model()
        embeddings = model.encode(
            [generated, expected],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity = dot product for L2-normalized vectors
        similarity = float(embeddings[0] @ embeddings[1])
        # Clamp to [0.0, 1.0] to handle floating point edge cases
        return round(max(0.0, min(1.0, similarity)), 4)
    except Exception as e:
        return 0.0


def semantic_similarity_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """
    Compute semantic similarity for a list of (generated, expected) pairs.
    Batched for efficiency -- one model.encode() call for all strings.

    Args:
        pairs: list of (generated_sql, expected_sql) tuples

    Returns:
        list of float similarity scores in same order as input
    """
    if not pairs:
        return []

    try:
        model   = _get_sem_model()
        strings = []
        for gen, exp in pairs:
            strings.append(gen or "")
            strings.append(exp or "")

        embeddings = model.encode(
            strings,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )

        scores = []
        for i in range(len(pairs)):
            gen_emb = embeddings[i * 2]
            exp_emb = embeddings[i * 2 + 1]
            sim     = float(gen_emb @ exp_emb)
            scores.append(round(max(0.0, min(1.0, sim)), 4))
        return scores

    except Exception as e:
        return [0.0] * len(pairs)


# -------------------------------------------------------
# Updated summarize with semantic similarity support
# -------------------------------------------------------

def summarize_with_semantics(
    results : list[ScoreResult],
    pairs   : list[tuple[str, str]] | None = None,
) -> dict:
    """
    Summarize results including average semantic similarity.

    Args:
        results : list of ScoreResult
        pairs   : list of (generated_sql, expected_sql) -- if provided,
                  computes semantic similarity for each pair

    Returns:
        dict with pass/partial/fail counts, rates, and avg_semantic_similarity
    """
    total   = len(results)
    passed  = sum(1 for r in results if r.verdict == "PASS")
    partial = sum(1 for r in results if r.verdict == "PARTIAL")
    failed  = sum(1 for r in results if r.verdict == "FAIL")

    summary = {
        "total"                  : total,
        "pass"                   : passed,
        "partial"                : partial,
        "fail"                   : failed,
        "pass_rate"              : round(passed  / total, 3) if total else 0,
        "partial_rate"           : round(partial / total, 3) if total else 0,
        "fail_rate"              : round(failed  / total, 3) if total else 0,
        "valid_rate"             : round((passed + partial) / total, 3) if total else 0,
        "avg_semantic_similarity": None,
    }

    if pairs:
        scores = semantic_similarity_batch(pairs)
        summary["avg_semantic_similarity"] = round(sum(scores) / len(scores), 4) if scores else None

    return summary


# -------------------------------------------------------
# Demo
# -------------------------------------------------------

if __name__ == "__main__":
    print("Metrics Demo")
    print("=" * 60)

    test_cases = [
        # (description, generated, expected)
        (
            "exact match",
            "SELECT Name FROM people WHERE Nationality != 'Russia'",
            "SELECT Name FROM people WHERE Nationality != 'Russia'",
        ),
        (
            "Week 1 gold_01: quote style difference",
            "SELECT Name FROM people WHERE Nationality != 'Russia'",
            'SELECT Name FROM people WHERE Nationality != "Russia"',
        ),
        (
            "Week 1 gold_03: added AS aliases",
            "SELECT MAX(Capacity) AS max_capacity, AVG(Average) AS avg FROM stadium",
            "SELECT MAX(capacity), average FROM stadium",
        ),
        (
            "Week 1 gold_04: COUNT(*) vs COUNT(col)",
            "SELECT country, COUNT(apid) FROM airports GROUP BY country ORDER BY COUNT(apid) DESC",
            "SELECT COUNT(*), country FROM airports GROUP BY country ORDER BY COUNT(*) DESC",
        ),
        (
            "Week 1 gold_05: INNER JOIN vs JOIN",
            "SELECT T2.Birth_Date FROM poker_player AS T1 INNER JOIN people AS T2 ON T1.People_ID = T2.People_ID ORDER BY T1.Earnings ASC LIMIT 1",
            "SELECT T1.Birth_Date FROM people AS T1 JOIN poker_player AS T2 ON T1.People_ID = T2.People_ID ORDER BY T2.Earnings ASC LIMIT 1",
        ),
        (
            "valid SQL but wrong",
            "SELECT name FROM users",
            "SELECT COUNT(*) FROM airports GROUP BY country",
        ),
        (
            "invalid SQL (gibberish)",
            "this is not sql at all",
            "SELECT name FROM users",
        ),
        (
            "empty generated",
            "",
            "SELECT name FROM users",
        ),
        (
            "intentional wrong pair (missing aggregate)",
            "SELECT * FROM AIRPORTS",
            "SELECT COUNT(*) FROM AIRPORTS",
        ),
    ]

    results = []
    for desc, generated, expected in test_cases:
        result = score(generated, expected)
        results.append(result)
        print(f"\n{desc}")
        print(f"  Generated : {generated[:70]}")
        print(f"  Expected  : {expected[:70]}")
        print(f"  {result}")
        if result.notes:
            print(f"  Notes     : {result.notes}")

    print(f"\n{'=' * 60}")
    summary = summarize(results)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")
    print("=" * 60)