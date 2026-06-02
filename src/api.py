"""
api.py -- Task 4: Flask REST API for the AI SQL Query Assistant
Endpoints:
    GET  /health     -- server status check
    POST /generate   -- NL -> SQL generation
    POST /evaluate   -- score a generated SQL against expected

Run:
    python src/api.py

NOTE: debug=True is for development only. Set to False before deployment.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Make src/ importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent))

from prompt_builder import PromptBuilder, VALID_STRATEGIES
from output_parser  import OutputParser
from metrics        import score as score_sql, valid_sql
from retriever      import Retriever

# -------------------------------------------------------
# App setup
# -------------------------------------------------------

app   = Flask(__name__)
CORS(app)  # Allow cross-origin requests for future frontend

MODEL       = "llama-3.3-70b-versatile"
TEMPERATURE = 0
MAX_TOKENS  = 300

# Shared instances — created once at startup
_parser    = OutputParser()
_client    = None   # lazy init so missing API key gives a clean error at request time
_retriever = None   # lazy init so missing embeddings give a clean error at request time


def get_retriever() -> Retriever:
    """Return Retriever, initializing lazily on first RAG request."""
    global _retriever
    if _retriever is None:
        try:
            _retriever = Retriever()
            print(f"[RAG] Retriever loaded: {_retriever.size} pairs indexed")
        except FileNotFoundError as e:
            raise RuntimeError(
                f"RAG retriever not available: {e}. "
                f"Run embed_pairs.py first."
            )
    return _retriever


def get_client() -> Groq:
    """Return Groq client, initializing lazily on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        _client = Groq(api_key=api_key)
    return _client


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def _call_groq(system_prompt: str, user_prompt: str) -> tuple[str, float]:
    """
    Call Groq API and return (raw_output, latency_seconds).
    Raises on API error.
    """
    client = get_client()
    t0 = time.time()
    response = client.chat.completions.create(
        model      = MODEL,
        temperature= TEMPERATURE,
        max_tokens = MAX_TOKENS,
        messages   = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    latency = round(time.time() - t0, 3)
    raw     = response.choices[0].message.content.strip()
    return raw, latency


def _error(message: str, status: int = 400) -> tuple:
    """Return a standard JSON error response."""
    return jsonify({"error": message}), status


# -------------------------------------------------------
# GET /health
# -------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Response:
        {
            "status"     : "ok",
            "model"      : "llama-3.3-70b-versatile",
            "strategies" : ["zero_shot", "few_shot", "chain_of_thought"],
            "timestamp"  : "2026-04-01T12:00:00"
        }
    """
    return jsonify({
        "status"    : "ok",
        "model"     : MODEL,
        "strategies": list(VALID_STRATEGIES),
        "timestamp" : datetime.now().isoformat(),
    })


# -------------------------------------------------------
# POST /generate
# -------------------------------------------------------

@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate SQL from a natural language query.

    Request body (JSON):
        {
            "nl_query"  : "How many airports do we have?",   -- required
            "schema"    : {                                   -- optional
                "tables" : ["airports"],
                "columns": ["id", "city", "country"]
            },
            "strategy"  : "zero_shot",                       -- optional, default: zero_shot
            "use_rag"   : false                              -- optional, default: false
        }

    Response:
        {
            "sql"            : "SELECT COUNT(*) FROM airports",
            "raw"            : "<raw LLM output before parsing>",
            "valid"          : true,
            "strategy"       : "zero_shot",
            "model"          : "llama-3.3-70b-versatile",
            "latency_s"      : 1.23,
            "parse_issues"   : [],
            "rag_used"       : false,
            "retrieved_pairs": []   -- populated when use_rag=true
        }

    Error responses:
        400 -- missing nl_query or invalid strategy
        500 -- API call failed
    """
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be JSON")

    # Validate required field
    nl_query = data.get("nl_query", "").strip()
    if not nl_query:
        return _error("nl_query is required and must not be empty")

    # Optional fields with defaults
    schema   = data.get("schema", {})
    strategy = data.get("strategy", "zero_shot")

    if strategy not in VALID_STRATEGIES:
        return _error(
            f"Invalid strategy '{strategy}'. "
            f"Valid options: {list(VALID_STRATEGIES)}"
        )

    # Ensure schema has the right keys
    if not isinstance(schema, dict):
        return _error("schema must be a JSON object with 'tables' and 'columns' keys")

    use_rag = data.get("use_rag", False)

    # RAG: retrieve similar examples and inject into prompt
    retrieved_pairs = []
    rag_examples    = []
    if use_rag:
        try:
            retriever       = get_retriever()
            retrieved       = retriever.retrieve(nl_query, k=3)
            retrieved_pairs = [r.to_dict() for r in retrieved]
            # Format as few-shot examples for prompt injection
            rag_examples    = [
                {"nl_query": r.nl_query, "sql_query": r.sql_query, "schema": {}}
                for r in retrieved
            ]
        except RuntimeError as e:
            return _error(str(e), 500)

    # Build prompt
    try:
        builder = PromptBuilder(strategy=strategy)
        if use_rag and rag_examples:
            # Override strategy to few_shot and inject RAG examples
            builder_rag       = PromptBuilder(strategy="few_shot")
            # Temporarily replace the safe_pairs with retrieved pairs
            builder_rag._safe_pairs = rag_examples
            prompt_result     = builder_rag.build(nl_query=nl_query, schema=schema)
        else:
            prompt_result = builder.build(nl_query=nl_query, schema=schema)
    except Exception as e:
        return _error(f"Prompt build failed: {str(e)}", 500)

    # Call Groq
    print(f"Generating SQL... strategy={strategy} nl={nl_query[:60]}")
    try:
        raw, latency = _call_groq(
            system_prompt=prompt_result.system_prompt,
            user_prompt  =prompt_result.prompt,
        )
    except RuntimeError as e:
        return _error(str(e), 500)
    except Exception as e:
        return _error(f"LLM API call failed: {str(e)}", 500)

    # Parse output
    parse_result = _parser.parse(raw)
    is_valid, _  = valid_sql(parse_result.sql)

    return jsonify({
        "sql"            : parse_result.sql,
        "raw"            : raw,
        "valid"          : is_valid,
        "strategy"       : strategy,
        "model"          : MODEL,
        "latency_s"      : latency,
        "parse_issues"   : parse_result.issues,
        "rag_used"       : use_rag,
        "retrieved_pairs": retrieved_pairs,
    })


# -------------------------------------------------------
# POST /evaluate
# -------------------------------------------------------

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Score a generated SQL against an expected SQL.
    Optionally generates SQL first if generated_sql is not provided.

    Request body (JSON):
        {
            "nl_query"      : "How many airports?",        -- required
            "expected_sql"  : "SELECT COUNT(*) FROM ...",  -- required
            "generated_sql" : "SELECT COUNT(*) FROM ...",  -- optional
            "schema"        : { "tables": [], "columns": [] }, -- optional
            "strategy"      : "zero_shot"                  -- optional, used if no generated_sql
        }

    Response:
        {
            "nl_query"      : "How many airports?",
            "expected_sql"  : "SELECT COUNT(*) FROM airports",
            "generated_sql" : "SELECT COUNT(*) FROM airports",
            "verdict"       : "PASS",
            "exact_match"   : true,
            "valid_sql"     : true,
            "score_notes"   : [],
            "latency_s"     : 1.23,    -- null if generated_sql was provided
            "strategy"      : "zero_shot"
        }

    Error responses:
        400 -- missing required fields
        500 -- generation or scoring failed
    """
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body must be JSON")

    nl_query     = data.get("nl_query", "").strip()
    expected_sql = data.get("expected_sql", "").strip()

    if not nl_query:
        return _error("nl_query is required")
    if not expected_sql:
        return _error("expected_sql is required")

    schema        = data.get("schema", {})
    strategy      = data.get("strategy", "zero_shot")
    generated_sql = data.get("generated_sql", "").strip()
    latency       = None

    # If no generated_sql provided, generate it first
    if not generated_sql:
        if strategy not in VALID_STRATEGIES:
            return _error(f"Invalid strategy '{strategy}'")

        try:
            builder       = PromptBuilder(strategy=strategy)
            prompt_result = builder.build(nl_query=nl_query, schema=schema)
        except Exception as e:
            return _error(f"Prompt build failed: {str(e)}", 500)

        print(f"Generating SQL for evaluate... nl={nl_query[:60]}")
        try:
            raw, latency = _call_groq(
                system_prompt=prompt_result.system_prompt,
                user_prompt  =prompt_result.prompt,
            )
        except Exception as e:
            return _error(f"LLM API call failed: {str(e)}", 500)

        parse_result  = _parser.parse(raw)
        generated_sql = parse_result.sql

    # Score
    try:
        result = score_sql(generated=generated_sql, expected=expected_sql)
    except Exception as e:
        return _error(f"Scoring failed: {str(e)}", 500)

    return jsonify({
        "nl_query"     : nl_query,
        "expected_sql" : expected_sql,
        "generated_sql": generated_sql,
        "verdict"      : result.verdict,
        "exact_match"  : result.exact,
        "valid_sql"    : result.valid,
        "score_notes"  : result.notes,
        "latency_s"    : latency,
        "strategy"     : strategy,
    })


# -------------------------------------------------------
# GET /metrics/summary
# -------------------------------------------------------

@app.route("/metrics/summary", methods=["GET"])
def metrics_summary():
    """
    Return aggregate evaluation stats from all result files.

    Response:
        {
            "baseline": { "pass": 8, "pass_rate": 0.12, "valid_rate": 0.91, ... },
            "rag"     : { "pass": 43, "pass_rate": 0.67, "valid_rate": 0.91, ... },
            "model_comparison": [ {...}, {...} ],
            "delta"   : { "pass_rate": 0.55, "semantic": 0.058 },
            "files_loaded": [...]
        }
    """
    import glob

    results_dir   = Path("outputs")
    files_loaded  = []
    baseline      = {}
    rag           = {}
    model_comp    = []

    # Load week3 baseline (64-pair, no RAG)
    w3_path = results_dir / "results_week3.json"
    if w3_path.exists():
        with open(w3_path, encoding="utf-8") as f:
            w3 = json.load(f)
        strats = w3.get("strategies", [])
        if strats:
            s = strats[0]["summary"]
            baseline = {
                "source"    : "results_week3.json",
                "n_pairs"   : s.get("total", 0),
                "pass"      : s.get("pass", 0),
                "pass_rate" : s.get("pass_rate", 0),
                "valid_rate": s.get("valid_rate", 0),
                "condition" : "baseline_no_rag",
            }
        files_loaded.append("results_week3.json")

    # Load week5 RAG full (64-pair, RAG)
    w5_path = results_dir / "results_week5_rag_full.json"
    if w5_path.exists():
        with open(w5_path, encoding="utf-8") as f:
            w5 = json.load(f)
        comp = w5.get("comparison", {})
        rag_summary = comp.get("rag", {}).get("summary", {})
        rag = {
            "source"     : "results_week5_rag_full.json",
            "n_pairs"    : rag_summary.get("total", 0),
            "pass"       : rag_summary.get("pass", 0),
            "pass_rate"  : rag_summary.get("pass_rate", 0),
            "valid_rate" : rag_summary.get("valid_rate", 0),
            "avg_semantic": rag_summary.get("avg_semantic", 0),
            "condition"  : "rag_k3",
        }
        files_loaded.append("results_week5_rag_full.json")

    # Load model comparison
    mc_path = results_dir / "model_comparison.json"
    if mc_path.exists():
        with open(mc_path, encoding="utf-8") as f:
            mc = json.load(f)
        for m in mc.get("models", []):
            s = m.get("summary", {})
            model_comp.append({
                "model"      : s.get("model", ""),
                "pass"       : s.get("pass", 0),
                "pass_rate"  : s.get("pass_rate", 0),
                "valid_rate" : s.get("valid_rate", 0),
                "avg_latency": s.get("avg_latency", 0),
                "avg_semantic": s.get("avg_semantic", 0),
            })
        files_loaded.append("model_comparison.json")

    # Compute delta
    delta = {}
    if baseline and rag:
        delta = {
            "pass_rate_delta": round(
                rag.get("pass_rate", 0) - baseline.get("pass_rate", 0), 3),
            "semantic_delta" : round(
                rag.get("avg_semantic", 0) - baseline.get("pass_rate", 0), 3),
        }

    return jsonify({
        "baseline"        : baseline,
        "rag"             : rag,
        "model_comparison": model_comp,
        "delta"           : delta,
        "files_loaded"    : files_loaded,
        "model"           : MODEL,
    })


# -------------------------------------------------------
# Error handlers
# -------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error"    : "Endpoint not found",
        "endpoints": [
            "GET  /health",
            "GET  /metrics/summary",
            "POST /generate",
            "POST /evaluate",
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  AI SQL Query Assistant API")
    print(f"  Model    : {MODEL}")
    print(f"  Strategies: {list(VALID_STRATEGIES)}")
    print("=" * 50)
    print("  Endpoints:")
    print("    GET  http://localhost:5000/health")
    print("    GET  http://localhost:5000/metrics/summary")
    print("    POST http://localhost:5000/generate")
    print("    POST http://localhost:5000/evaluate")
    print("=" * 50)
    # NOTE: debug=True is for development only. Set to False in production.
    app.run(host="0.0.0.0", port=5000, debug=True)