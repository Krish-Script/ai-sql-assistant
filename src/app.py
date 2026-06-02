"""
app.py -- Week 5: Streamlit UI for the AI SQL Query Assistant
Requires: Flask API running at localhost:5000 (python src/api.py)

Run:
    streamlit run src/app.py

Tabs:
    Generate  -- NL -> SQL with strategy selector, RAG toggle, results panel
    Evaluate  -- Run golden set, show metrics table
    History   -- Accumulated session queries with export
"""

import json
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from pathlib import Path

# -------------------------------------------------------
# Config
# -------------------------------------------------------

API_BASE    = "http://localhost:5000"
TIMEOUT_SEC = 30
GOLDEN_PATH = Path("data/golden/golden.json")

STRATEGY_LABELS = {
    "zero_shot"       : "Zero-Shot",
    "few_shot"        : "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
}

VERDICT_ICONS = {
    "PASS"   : "PASS",
    "PARTIAL": "PARTIAL",
    "FAIL"   : "FAIL",
}

# -------------------------------------------------------
# Page config
# -------------------------------------------------------

st.set_page_config(
    page_title = "AI SQL Assistant",
    page_icon  = "🛢",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# -------------------------------------------------------
# Session state init
# -------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# -------------------------------------------------------
# API helpers
# -------------------------------------------------------

def api_health():
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_generate(nl_query, strategy, schema, use_rag):
    try:
        resp = requests.post(
            f"{API_BASE}/generate",
            json    = {"nl_query": nl_query, "strategy": strategy,
                       "schema": schema, "use_rag": use_rag},
            timeout = TIMEOUT_SEC,
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.json().get("error", "Unknown")}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is python src/api.py running?"}
    except Exception as e:
        return {"error": str(e)}


def api_evaluate(nl_query, expected_sql, generated_sql):
    try:
        resp = requests.post(
            f"{API_BASE}/evaluate",
            json    = {"nl_query": nl_query, "expected_sql": expected_sql,
                       "generated_sql": generated_sql},
            timeout = TIMEOUT_SEC,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

with st.sidebar:
    st.title("🛢 AI SQL Assistant")
    st.caption("RAG-powered NL to SQL generation")
    st.divider()

    health = api_health()
    if health:
        st.success(f"API online")
        st.caption(f"Model: {health['model']}")
    else:
        st.error("API offline")
        st.caption("Run: python src/api.py")

    st.divider()

    strategy_key = st.selectbox(
        "Prompt Strategy",
        options     = list(STRATEGY_LABELS.keys()),
        format_func = lambda k: STRATEGY_LABELS[k],
        help        = "zero_shot: fastest | few_shot: balanced | chain_of_thought: most reasoning",
    )

    use_rag = st.toggle(
        "Enable RAG",
        value = False,
        help  = "Retrieves top-3 similar NL->SQL examples to guide generation",
    )

    if use_rag:
        st.info("RAG active: top-3 similar pairs injected into prompt")

    st.divider()
    session_count_placeholder = st.empty()
    session_count_placeholder.caption(f"Session queries: {len(st.session_state.history)}")


# -------------------------------------------------------
# Tabs
# -------------------------------------------------------

tab_generate, tab_evaluate, tab_history = st.tabs([
    "Generate", "Evaluate", "History"
])


# ===============================================================
# TAB 1 -- GENERATE
# ===============================================================

with tab_generate:
    st.header("Generate SQL from Natural Language")

    with st.expander("Optional: provide database schema hint", expanded=False):
        col_t, col_c = st.columns(2)
        with col_t:
            tables_input = st.text_input(
                "Tables (comma-separated)",
                placeholder="airports, flights, airlines",
            )
        with col_c:
            columns_input = st.text_input(
                "Columns (comma-separated)",
                placeholder="airport_id, city, country",
            )

    schema = {}
    if tables_input or columns_input:
        schema = {
            "tables" : [t.strip() for t in tables_input.split(",") if t.strip()],
            "columns": [c.strip() for c in columns_input.split(",") if c.strip()],
        }

    nl_query = st.text_area(
        "Natural language question",
        placeholder = "e.g. How many airports are there per country, ordered from most to least?",
        height      = 100,
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        generate_clicked = st.button("Generate SQL", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear"):
            st.rerun()

    if generate_clicked:
        if not nl_query.strip():
            st.warning("Please enter a natural language question.")
        elif not health:
            st.error("API is offline. Start the server: python src/api.py")
        else:
            with st.spinner("Generating SQL..."):
                result = api_generate(
                    nl_query = nl_query.strip(),
                    strategy = strategy_key,
                    schema   = schema,
                    use_rag  = use_rag,
                )

            if result and "error" not in result:
                st.divider()
                st.subheader("Result")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Valid SQL", "Yes" if result.get("valid") else "No")
                m2.metric("Strategy", STRATEGY_LABELS.get(result.get("strategy",""), result.get("strategy","")))
                m3.metric("RAG", "Enabled" if result.get("rag_used") else "Disabled")
                m4.metric("Latency", f"{result.get('latency_s', 0):.2f}s")

                # SQL output
                st.subheader("Generated SQL")
                sql_out = result.get("sql", "")
                st.code(sql_out, language="sql")

                # Parse issues
                if result.get("parse_issues"):
                    st.caption(f"Parser fixes applied: {', '.join(result['parse_issues'])}")

                # RAG panel
                if result.get("rag_used") and result.get("retrieved_pairs"):
                    st.divider()
                    st.subheader("Retrieved Examples (RAG)")
                    st.caption("Top-3 most similar NL->SQL pairs used to guide generation")
                    for i, pair in enumerate(result["retrieved_pairs"], 1):
                        with st.expander(
                            f"[{i}] sim={pair['similarity']:.4f}  |  {pair['nl_query'][:70]}",
                            expanded = i == 1,
                        ):
                            st.write(f"**Question:** {pair['nl_query']}")
                            st.code(pair["sql_query"], language="sql")
                            c1, c2, c3 = st.columns(3)
                            c1.caption(f"Scenario: {pair['scenario']}")
                            c2.caption(f"Difficulty: {pair['difficulty']}")
                            c3.caption(f"Source: {pair['source']}")

                # Save to history
                st.session_state.history.append({
                    "timestamp" : datetime.now().strftime("%H:%M:%S"),
                    "nl_query"  : nl_query.strip(),
                    "sql"       : sql_out,
                    "strategy"  : result.get("strategy", ""),
                    "rag_used"  : result.get("rag_used", False),
                    "valid"     : result.get("valid", False),
                    "latency_s" : result.get("latency_s", 0),
                })
                # Update session count in sidebar immediately
                session_count_placeholder.caption(
                    f"Session queries: {len(st.session_state.history)}"
                )

            elif result and "error" in result:
                st.error(f"Error: {result['error']}")


# ===============================================================
# TAB 2 -- EVALUATE
# ===============================================================

with tab_evaluate:
    st.header("Evaluate on Golden Set")
    st.caption("Runs all 10 correct golden pairs through the API and scores each result.")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        eval_rag = st.toggle("Use RAG for eval", value=False)
        run_eval = st.button("Run Evaluation", type="primary", use_container_width=True)
    with col_info:
        st.info(
            "Runs 10 API calls (~30s). "
            "Scores each pair using exact match + valid SQL. "
            "RAG toggle uses retrieved examples for each query."
        )

    if run_eval:
        if not GOLDEN_PATH.exists():
            st.error(f"Golden set not found at {GOLDEN_PATH}. Run golden.py first.")
        elif not health:
            st.error("API is offline.")
        else:
            with open(GOLDEN_PATH, encoding="utf-8") as f:
                golden = json.load(f)

            correct_pairs = [g for g in golden if g.get("is_correct", True)]
            results       = []
            progress_bar  = st.progress(0, text="Running evaluation...")

            for i, pair in enumerate(correct_pairs):
                progress_bar.progress(
                    (i + 1) / len(correct_pairs),
                    text=f"Pair {i+1}/{len(correct_pairs)}: {pair['nl_query'][:50]}..."
                )
                gen = api_generate(
                    nl_query = pair["nl_query"],
                    strategy = strategy_key,
                    schema   = pair.get("schema", {}),
                    use_rag  = eval_rag,
                )
                if gen and "error" not in gen:
                    score = api_evaluate(
                        nl_query     = pair["nl_query"],
                        expected_sql = pair["sql_query"],
                        generated_sql= gen.get("sql", ""),
                    )
                    verdict = score.get("verdict", "FAIL") if score else "FAIL"
                else:
                    verdict = "FAIL"
                    gen     = {}

                results.append({
                    "gold_id"   : pair.get("gold_id", f"pair_{i}"),
                    "scenario"  : pair.get("scenario", ""),
                    "difficulty": pair.get("difficulty", ""),
                    "nl_query"  : pair["nl_query"][:60],
                    "verdict"   : verdict,
                    "valid_sql" : gen.get("valid", False),
                    "latency_s" : gen.get("latency_s", 0),
                })
                time.sleep(0.3)

            progress_bar.empty()
            st.session_state.eval_results = results

    if st.session_state.eval_results:
        results = st.session_state.eval_results
        st.divider()
        st.subheader("Evaluation Results")

        # Summary metrics
        total   = len(results)
        passed  = sum(1 for r in results if r["verdict"] == "PASS")
        partial = sum(1 for r in results if r["verdict"] == "PARTIAL")
        failed  = sum(1 for r in results if r["verdict"] == "FAIL")
        valid   = sum(1 for r in results if r["valid_sql"])
        avg_lat = sum(r["latency_s"] for r in results) / total if total else 0

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PASS",    f"{passed}/{total}",  f"{passed/total*100:.0f}%")
        m2.metric("PARTIAL", f"{partial}/{total}", f"{partial/total*100:.0f}%")
        m3.metric("FAIL",    f"{failed}/{total}",  f"{failed/total*100:.0f}%")
        m4.metric("Valid SQL", f"{valid}/{total}",  f"{valid/total*100:.0f}%")
        m5.metric("Avg Latency", f"{avg_lat:.2f}s")

        st.divider()

        # Results table
        df = pd.DataFrame(results)
        def color_verdict(val):
            if val == "PASS":
                return "background-color: #d4edda; color: #155724"
            elif val == "PARTIAL":
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"

        styled = df.style.applymap(color_verdict, subset=["verdict"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Scenario breakdown
        st.divider()
        st.subheader("Scenario Breakdown")
        scenario_df = df.groupby("scenario")["verdict"].value_counts().unstack(fill_value=0)
        st.dataframe(scenario_df, use_container_width=True)


# ===============================================================
# TAB 3 -- HISTORY
# ===============================================================

with tab_history:
    st.header("Session History")

    if not st.session_state.history:
        st.info("No queries yet. Use the Generate tab to start.")
    else:
        col_exp, col_dl = st.columns([1, 1])
        with col_exp:
            if st.button("Clear history"):
                st.session_state.history = []
                st.rerun()
        with col_dl:
            history_json = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                label     = "Export history (JSON)",
                data      = history_json,
                file_name = f"sql_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime      = "application/json",
            )

        st.divider()

        for i, entry in enumerate(reversed(st.session_state.history), 1):
            rag_tag = " | RAG" if entry.get("rag_used") else ""
            valid_tag = "Valid" if entry.get("valid") else "Invalid"
            with st.expander(
                f"[{entry['timestamp']}]  {entry['nl_query'][:70]}  |  {valid_tag}{rag_tag}",
                expanded = i == 1,
            ):
                st.write(f"**Question:** {entry['nl_query']}")
                st.code(entry["sql"], language="sql")
                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"Strategy: {STRATEGY_LABELS.get(entry['strategy'], entry['strategy'])}")
                c2.caption(f"RAG: {'Yes' if entry.get('rag_used') else 'No'}")
                c3.caption(f"Valid: {'Yes' if entry.get('valid') else 'No'}")
                c4.caption(f"Latency: {entry.get('latency_s', 0):.2f}s")