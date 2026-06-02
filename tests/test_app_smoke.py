"""
tests/test_app_smoke.py -- Task 8: Streamlit component existence checks
Verifies app.py imports cleanly and contains required UI components.
Does NOT require the Streamlit server or Flask API to be running.

Run:
    pytest tests/test_app_smoke.py -v
"""

import ast
import sys
import pytest
from pathlib import Path

APP_PATH = Path("src/app.py")


# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def load_app_source() -> str:
    assert APP_PATH.exists(), f"app.py not found at {APP_PATH}"
    return APP_PATH.read_text(encoding="utf-8")


def parse_app_ast() -> ast.Module:
    source = load_app_source()
    return ast.parse(source)


def get_all_calls(tree: ast.Module) -> list[str]:
    """Extract all function call names from the AST."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    return calls


def get_string_literals(tree: ast.Module) -> list[str]:
    """Extract all string literals from the AST."""
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    return strings


# -------------------------------------------------------
# Tests
# -------------------------------------------------------

class TestAppStructure:
    """Verify app.py has correct structure and imports."""

    def test_app_file_exists(self):
        assert APP_PATH.exists(), "src/app.py not found"

    def test_app_parses_without_syntax_error(self):
        source = load_app_source()
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"app.py has syntax error: {e}")

    def test_app_imports_streamlit(self):
        source = load_app_source()
        assert "import streamlit as st" in source, "app.py must import streamlit"

    def test_app_imports_requests(self):
        source = load_app_source()
        assert "import requests" in source, "app.py must import requests for API calls"

    def test_app_imports_pandas(self):
        source = load_app_source()
        assert "import pandas as pd" in source, "app.py must import pandas for results table"


class TestRequiredComponents:
    """Verify all required Streamlit components are present."""

    def test_has_three_tabs(self):
        source = load_app_source()
        assert "st.tabs" in source, "app.py must use st.tabs for navigation"

    def test_has_generate_tab(self):
        source = load_app_source()
        assert "Generate" in source, "app.py must have a Generate tab"

    def test_has_evaluate_tab(self):
        source = load_app_source()
        assert "Evaluate" in source, "app.py must have an Evaluate tab"

    def test_has_history_tab(self):
        source = load_app_source()
        assert "History" in source, "app.py must have a History tab"

    def test_has_rag_toggle(self):
        source = load_app_source()
        assert "use_rag" in source, "app.py must have a RAG toggle"
        assert "st.toggle" in source, "app.py must use st.toggle for RAG"

    def test_has_strategy_selector(self):
        source = load_app_source()
        assert "st.selectbox" in source, "app.py must have strategy selectbox"
        assert "strategy" in source.lower(), "app.py must reference strategy"

    def test_has_generate_button(self):
        source = load_app_source()
        assert "Generate SQL" in source, "app.py must have Generate SQL button"

    def test_has_sql_code_display(self):
        tree  = parse_app_ast()
        calls = get_all_calls(tree)
        assert "code" in calls, "app.py must use st.code to display SQL output"

    def test_has_metrics_display(self):
        tree  = parse_app_ast()
        calls = get_all_calls(tree)
        assert "metric" in calls, "app.py must use st.metric for stats display"

    def test_has_health_check(self):
        source = load_app_source()
        assert "api_health" in source, "app.py must check API health"
        assert "/health" in source, "app.py must call /health endpoint"


class TestRAGComponents:
    """Verify RAG-specific UI components are present."""

    def test_has_retrieved_pairs_panel(self):
        source = load_app_source()
        assert "retrieved_pairs" in source, "app.py must show retrieved RAG pairs"

    def test_has_similarity_score_display(self):
        source = load_app_source()
        assert "similarity" in source, "app.py must display similarity scores"

    def test_has_rag_expander(self):
        source = load_app_source()
        assert "st.expander" in source, "app.py must use expanders for RAG pairs"

    def test_rag_uses_api_use_rag_param(self):
        source = load_app_source()
        assert '"use_rag"' in source or "'use_rag'" in source, \
            "app.py must pass use_rag to the API"


class TestHistoryAndExport:
    """Verify history accumulation and export are present."""

    def test_has_session_state_history(self):
        source = load_app_source()
        assert "st.session_state.history" in source, \
            "app.py must use session_state to store history"

    def test_has_download_button(self):
        source = load_app_source()
        assert "st.download_button" in source, \
            "app.py must have a download button for history export"

    def test_has_export_json(self):
        source = load_app_source()
        assert "json.dumps" in source, \
            "app.py must serialize history to JSON for export"

    def test_history_includes_rag_flag(self):
        source = load_app_source()
        assert '"rag_used"' in source or "'rag_used'" in source, \
            "app.py must store rag_used in history entries"


class TestAPIIntegration:
    """Verify app.py integrates correctly with Flask API endpoints."""

    def test_calls_generate_endpoint(self):
        source = load_app_source()
        assert "/generate" in source, "app.py must call /generate endpoint"

    def test_calls_evaluate_endpoint(self):
        source = load_app_source()
        assert "/evaluate" in source, "app.py must call /evaluate endpoint"

    def test_calls_health_endpoint(self):
        source = load_app_source()
        assert "/health" in source, "app.py must call /health endpoint"

    def test_has_connection_error_handling(self):
        source = load_app_source()
        assert "ConnectionError" in source, \
            "app.py must handle requests.exceptions.ConnectionError"

    def test_has_timeout_config(self):
        source = load_app_source()
        assert "TIMEOUT_SEC" in source or "timeout" in source, \
            "app.py must set a request timeout"