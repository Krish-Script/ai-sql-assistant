"""
tests/test_api.py -- Task 7: API endpoint tests using pytest + requests
Tests run against the live Flask server at localhost:5000.

IMPORTANT: Start the API server before running these tests:
    python src/api.py

Run tests:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short   # shorter tracebacks

Test groups:
    TestHealth      -- GET /health (2 tests)
    TestGenerate    -- POST /generate (7 tests)
    TestEvaluate    -- POST /evaluate (6 tests)
    TestErrors      -- error handling across endpoints (4 tests)
"""

import pytest
import requests

BASE_URL = "http://localhost:5000"
TIMEOUT  = 30   # LLM calls can take up to 8s


# -------------------------------------------------------
# Fixtures
# -------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Verify the API server is running before any tests execute."""
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200, (
            f"Server returned {resp.status_code}. "
            f"Make sure api.py is running: python src/api.py"
        )
    except requests.exceptions.ConnectionError:
        pytest.exit(
            f"Cannot connect to {BASE_URL}. "
            f"Start the server first: python src/api.py",
            returncode=1,
        )


@pytest.fixture
def simple_schema():
    return {
        "tables" : ["airports"],
        "columns": ["airport_id", "city", "country", "airport_name"],
    }


@pytest.fixture
def join_schema():
    return {
        "tables" : ["customers", "orders"],
        "columns": ["customer_id", "name", "email", "order_id", "amount", "region"],
    }


# -------------------------------------------------------
# TestHealth
# -------------------------------------------------------

class TestHealth:
    """Tests for GET /health"""

    def test_health_returns_200(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        assert resp.status_code == 200

    def test_health_response_schema(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        assert "status"     in data
        assert "model"      in data
        assert "strategies" in data
        assert "timestamp"  in data
        assert data["status"] == "ok"

    def test_health_has_all_strategies(self):
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        data = resp.json()
        strategies = data["strategies"]
        assert "zero_shot"       in strategies
        assert "few_shot"        in strategies
        assert "chain_of_thought" in strategies


# -------------------------------------------------------
# TestGenerate
# -------------------------------------------------------

class TestGenerate:
    """Tests for POST /generate"""

    def test_generate_returns_200(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "How many airports do we have?",
                      "schema": simple_schema},
            timeout= TIMEOUT,
        )
        assert resp.status_code == 200

    def test_generate_response_has_required_fields(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "Count all airports", "schema": simple_schema},
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert "sql"          in data
        assert "raw"          in data
        assert "valid"        in data
        assert "strategy"     in data
        assert "model"        in data
        assert "latency_s"    in data
        assert "parse_issues" in data

    def test_generate_returns_valid_sql(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "How many airports do we have?",
                      "schema": simple_schema},
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert data["valid"] is True
        assert "SELECT" in data["sql"].upper()

    def test_generate_default_strategy_is_zero_shot(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "Count airports", "schema": simple_schema},
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert data["strategy"] == "zero_shot"

    def test_generate_respects_strategy_param(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "Count airports",
                      "schema"  : simple_schema,
                      "strategy": "few_shot"},
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert data["strategy"] == "few_shot"

    def test_generate_latency_is_reasonable(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "List all airports", "schema": simple_schema},
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert data["latency_s"] <= 10.0, (
            f"Latency {data['latency_s']}s exceeds 10s threshold"
        )

    def test_generate_works_without_schema(self):
        """Schema is optional -- API should still work."""
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "Count all users"},
            timeout= TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sql" in data


# -------------------------------------------------------
# TestEvaluate
# -------------------------------------------------------

class TestEvaluate:
    """Tests for POST /evaluate"""

    def test_evaluate_returns_200(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"    : "How many airports do we have?",
                "expected_sql": "SELECT COUNT(*) FROM airports",
                "schema"      : simple_schema,
            },
            timeout= TIMEOUT,
        )
        assert resp.status_code == 200

    def test_evaluate_response_has_required_fields(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"    : "How many airports?",
                "expected_sql": "SELECT COUNT(*) FROM airports",
                "schema"      : simple_schema,
            },
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert "nl_query"      in data
        assert "expected_sql"  in data
        assert "generated_sql" in data
        assert "verdict"       in data
        assert "exact_match"   in data
        assert "valid_sql"     in data
        assert "score_notes"   in data

    def test_evaluate_verdict_is_valid_value(self, simple_schema):
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"    : "How many airports?",
                "expected_sql": "SELECT COUNT(*) FROM airports",
                "schema"      : simple_schema,
            },
            timeout= TIMEOUT,
        )
        data = resp.json()
        assert data["verdict"] in ("PASS", "PARTIAL", "FAIL")

    def test_evaluate_accepts_pregenerated_sql(self):
        """Should score without calling the LLM if generated_sql is provided."""
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"     : "How many airports?",
                "expected_sql" : "SELECT COUNT(*) FROM airports",
                "generated_sql": "SELECT COUNT(*) FROM airports",
            },
            timeout= 5,   # no LLM call so should be fast
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"]     == "PASS"
        assert data["exact_match"] == True
        assert data["latency_s"]   is None   # no LLM call made

    def test_evaluate_detects_wrong_sql(self):
        """The intentional wrong pair -- missing aggregate."""
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"     : "How many airports do we have?",
                "expected_sql" : "SELECT COUNT(*) FROM airports",
                "generated_sql": "SELECT * FROM airports",
            },
            timeout= 5,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict"]     != "PASS"
        assert data["exact_match"] == False

    def test_evaluate_pass_on_exact_match(self):
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {
                "nl_query"     : "Find all users",
                "expected_sql" : "SELECT * FROM users",
                "generated_sql": "SELECT * FROM users",
            },
            timeout= 5,
        )
        data = resp.json()
        assert data["verdict"]     == "PASS"
        assert data["exact_match"] == True


# -------------------------------------------------------
# TestErrors
# -------------------------------------------------------

class TestErrors:
    """Tests for error handling across endpoints."""

    def test_generate_missing_nl_query_returns_400(self):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"strategy": "zero_shot"},
            timeout= 5,
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_generate_empty_nl_query_returns_400(self):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": ""},
            timeout= 5,
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_generate_invalid_strategy_returns_400(self):
        resp = requests.post(
            f"{BASE_URL}/generate",
            json   = {"nl_query": "Count airports", "strategy": "invalid_strategy"},
            timeout= 5,
        )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert "invalid_strategy" in data["error"]

    def test_evaluate_missing_expected_sql_returns_400(self):
        resp = requests.post(
            f"{BASE_URL}/evaluate",
            json   = {"nl_query": "How many airports?"},
            timeout= 5,
        )
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_unknown_endpoint_returns_404(self):
        resp = requests.get(f"{BASE_URL}/nonexistent", timeout=5)
        assert resp.status_code == 404
        data = resp.json()
        assert "error"     in data
        assert "endpoints" in data