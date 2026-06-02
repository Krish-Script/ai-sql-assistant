"""
prompt_builder.py -- Task 1: Build prompts for NL->SQL generation
Used by: eval_runner.py

3 strategies:
    zero_shot       : schema + question, no examples
    few_shot        : 3 examples from pairs_clean.json + schema + question
    chain_of_thought: ask model to reason step by step before writing SQL

Key rule: few-shot examples NEVER come from the golden set.
The builder enforces this by loading from pairs_clean.json and
filtering out any pair whose nl_query matches a golden set entry.

Usage:
    builder = PromptBuilder(strategy="few_shot")
    prompt  = builder.build(
        nl_query="How many airports do we have?",
        schema={"tables": ["airports"], "columns": ["id", "city", "country"]}
    )
    print(prompt)
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass


# -------------------------------------------------------
# Config
# -------------------------------------------------------

PAIRS_PATH  = Path("data/processed/pairs_clean.json")
GOLDEN_PATH = Path("data/golden/golden.json")

VALID_STRATEGIES = ("zero_shot", "few_shot", "chain_of_thought")

# Number of few-shot examples to include
N_FEW_SHOT = 3


# -------------------------------------------------------
# PromptResult
# -------------------------------------------------------

@dataclass
class PromptResult:
    """
    Result of building a prompt.

    Attributes:
        strategy      : which strategy was used
        prompt        : the full prompt string to send to the LLM
        system_prompt : the system message
        examples_used : list of nl_query strings used as few-shot examples
    """
    strategy     : str
    prompt       : str
    system_prompt: str
    examples_used: list[str]

    def __str__(self) -> str:
        lines = [
            f"Strategy : {self.strategy}",
            f"Examples : {len(self.examples_used)}",
            f"Prompt   :\n{self.prompt[:300]}{'...' if len(self.prompt) > 300 else ''}",
        ]
        return "\n".join(lines)


# -------------------------------------------------------
# PromptBuilder
# -------------------------------------------------------

class PromptBuilder:
    """
    Builds prompts for NL->SQL generation using one of 3 strategies.

    Args:
        strategy     : "zero_shot" | "few_shot" | "chain_of_thought"
        pairs_path   : path to pairs_clean.json (source of few-shot examples)
        golden_path  : path to golden.json (examples to exclude from few-shot)
        n_few_shot   : number of few-shot examples to include (default 3)
        seed         : random seed for reproducible example selection
    """

    def __init__(
        self,
        strategy   : str = "zero_shot",
        pairs_path : str | Path = PAIRS_PATH,
        golden_path: str | Path = GOLDEN_PATH,
        n_few_shot : int = N_FEW_SHOT,
        seed       : int = 42,
    ):
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Choose from: {VALID_STRATEGIES}"
            )
        self.strategy    = strategy
        self.n_few_shot  = n_few_shot
        self.seed        = seed
        self._pairs      = self._load_pairs(Path(pairs_path))
        self._golden_nls = self._load_golden_nls(Path(golden_path))

        # Pre-filter: remove any pair whose NL is in the golden set
        self._safe_pairs = [
            p for p in self._pairs
            if p["nl_query"] not in self._golden_nls
        ]

    # -------------------------------------------------------
    # Public API
    # -------------------------------------------------------

    def build(self, nl_query: str, schema: dict) -> PromptResult:
        """
        Build a prompt for the given NL query and schema.

        Args:
            nl_query : natural language question
            schema   : dict with keys 'tables' and 'columns'

        Returns:
            PromptResult with system_prompt, prompt, and metadata
        """
        schema_text = self._format_schema(schema)

        if self.strategy == "zero_shot":
            return self._zero_shot(nl_query, schema_text)
        elif self.strategy == "few_shot":
            return self._few_shot(nl_query, schema_text)
        elif self.strategy == "chain_of_thought":
            return self._chain_of_thought(nl_query, schema_text)

    def available_strategies(self) -> tuple:
        return VALID_STRATEGIES

    # -------------------------------------------------------
    # Strategy implementations
    # -------------------------------------------------------

    def _zero_shot(self, nl_query: str, schema_text: str) -> PromptResult:
        """
        Strategy 1: Zero-shot
        Give the model the schema and question with clear instructions.
        No examples. Tests raw model capability.
        """
        system = (
            "You are an expert SQL assistant. "
            "Given a database schema and a natural language question, "
            "return ONLY the SQL query. "
            "Do not explain. Do not use markdown. "
            "Do not add any text before or after the SQL. "
            "Use DISTINCT when the question implies unique or different values. "
            "Use foreign key relationships between tables when writing JOINs."
        )

        prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {nl_query}\n\n"
            f"SQL:"
        )

        return PromptResult(
            strategy     = "zero_shot",
            prompt       = prompt,
            system_prompt= system,
            examples_used= [],
        )

    def _few_shot(self, nl_query: str, schema_text: str) -> PromptResult:
        """
        Strategy 2: Few-shot
        Include 3 diverse NL->SQL examples before the actual question.
        Examples are selected to cover different scenarios.
        NEVER uses golden set examples.
        """
        system = (
            "You are an expert SQL assistant. "
            "Study the examples below, then generate SQL for the new question. "
            "Return ONLY the SQL query. "
            "Do not explain. Do not use markdown. "
            "Do not add any text before or after the SQL. "
            "Use DISTINCT when the question implies unique or different values. "
            "Use foreign key relationships between tables when writing JOINs."
        )

        examples = self._select_few_shot_examples()
        examples_text = self._format_examples(examples)

        prompt = (
            f"{examples_text}\n"
            f"-- Now answer the following:\n\n"
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {nl_query}\n\n"
            f"SQL:"
        )

        return PromptResult(
            strategy     = "few_shot",
            prompt       = prompt,
            system_prompt= system,
            examples_used= [e["nl_query"] for e in examples],
        )

    def _chain_of_thought(self, nl_query: str, schema_text: str) -> PromptResult:
        """
        Strategy 3: Chain-of-thought
        Ask the model to reason about tables and columns before writing SQL.
        Uses a structured thinking format to reduce hallucination.
        """
        system = (
            "You are a SQL expert. "
            "Think through the query step by step, then write the final SQL. "
            "Rules: Use DISTINCT when the question implies uniqueness. "
            "Use foreign key relationships between tables when writing JOINs. "
            "Always include all relevant columns. "
            "Your final answer MUST start with SELECT and contain only SQL — "
            "no explanation after the query."
        )

        prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"Question: {nl_query}\n\n"
            f"Think step by step:\n"
            f"1. Which tables do I need?\n"
            f"2. Which columns are relevant?\n"
            f"3. What SQL pattern fits (filter/join/aggregate/subquery)?\n"
            f"4. Does the question imply DISTINCT?\n\n"
            f"Now write ONLY the SQL query. Start with SELECT:"
        )

        return PromptResult(
            strategy     = "chain_of_thought",
            prompt       = prompt,
            system_prompt= system,
            examples_used= [],
        )

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def _format_schema(self, schema: dict) -> str:
        """Format schema dict into a readable string for the prompt."""
        tables  = schema.get("tables",  [])
        columns = schema.get("columns", [])

        if not tables and not columns:
            return "No schema provided."

        lines = []
        if tables:
            lines.append(f"Tables : {', '.join(tables)}")
        if columns:
            # Cap at 15 columns to keep prompt short
            shown = columns[:15]
            extra = len(columns) - 15
            col_str = ", ".join(shown)
            if extra > 0:
                col_str += f" ... (+{extra} more)"
            lines.append(f"Columns: {col_str}")

        return "\n".join(lines)

    def _select_few_shot_examples(self) -> list[dict]:
        """
        Select n_few_shot diverse examples from safe pairs.
        Tries to cover different scenarios for better diversity.
        Never overlaps with the golden set.
        """
        if not self._safe_pairs:
            return []

        # Group by scenario for diversity
        by_scenario: dict[str, list] = {}
        for p in self._safe_pairs:
            sc = p.get("scenario", "other")
            by_scenario.setdefault(sc, []).append(p)

        # Pick one from different scenarios, cycling through them
        rng      = random.Random(self.seed)
        selected = []
        scenarios = list(by_scenario.keys())
        rng.shuffle(scenarios)

        for sc in scenarios:
            if len(selected) >= self.n_few_shot:
                break
            candidates = by_scenario[sc]
            # Prefer easy/medium difficulty for clearer examples
            easy = [p for p in candidates if p.get("difficulty") in ("easy", "medium")]
            pool = easy if easy else candidates
            selected.append(rng.choice(pool))

        # Fill remaining slots if needed
        remaining = self.n_few_shot - len(selected)
        if remaining > 0:
            used_nls = {p["nl_query"] for p in selected}
            extras   = [p for p in self._safe_pairs if p["nl_query"] not in used_nls]
            rng.shuffle(extras)
            selected.extend(extras[:remaining])

        return selected[:self.n_few_shot]

    def _format_examples(self, examples: list[dict]) -> str:
        """Format few-shot examples as a prompt string."""
        lines = ["-- Examples:\n"]
        for i, ex in enumerate(examples, start=1):
            schema_text = self._format_schema(ex.get("schema", {}))
            lines.append(f"-- Example {i}:")
            lines.append(f"Database schema:\n{schema_text}")
            lines.append(f"Question: {ex['nl_query']}")
            lines.append(f"SQL: {ex['sql_query']}")
            lines.append("")
        return "\n".join(lines)

    def _load_pairs(self, path: Path) -> list[dict]:
        """Load pairs_clean.json. Returns empty list if file not found."""
        if not path.exists():
            print(f"[PromptBuilder] Warning: {path} not found. Few-shot disabled.")
            return []
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_golden_nls(self, path: Path) -> set[str]:
        """Load golden set NL queries to exclude from few-shot examples."""
        if not path.exists():
            return set()
        with open(path, encoding="utf-8") as f:
            golden = json.load(f)
        return {g["nl_query"] for g in golden}


# -------------------------------------------------------
# Demo
# -------------------------------------------------------

if __name__ == "__main__":
    # Sample schema and query for demo
    sample_schema = {
        "tables" : ["airports", "flights", "airlines"],
        "columns": ["airport_id", "city", "country", "airport_name",
                    "flight_id", "source_airport", "dest_airport",
                    "airline_id", "airline_name"],
    }
    sample_nl = "What is the number of airports per country, ordered from most to least?"

    print("PromptBuilder Demo")
    print("=" * 65)

    for strategy in VALID_STRATEGIES:
        builder = PromptBuilder(strategy=strategy)
        result  = builder.build(nl_query=sample_nl, schema=sample_schema)

        print(f"\n[Strategy: {strategy.upper()}]")
        print(f"System : {result.system_prompt[:80]}...")
        if result.examples_used:
            print(f"Examples used ({len(result.examples_used)}):")
            for ex in result.examples_used:
                print(f"  - {ex[:60]}")
        print(f"\nPrompt preview (first 400 chars):")
        print("-" * 40)
        print(result.prompt[:400])
        print("-" * 40)

    print("\nAll 3 strategies built successfully!")

    # Verify golden set exclusion
    builder  = PromptBuilder(strategy="few_shot")
    golden   = builder._golden_nls
    safe_nls = {p["nl_query"] for p in builder._safe_pairs}
    overlap  = golden & safe_nls
    print(f"\nGolden set exclusion check:")
    print(f"  Golden NLs    : {len(golden)}")
    print(f"  Safe pairs    : {len(builder._safe_pairs)}")
    print(f"  Overlap (must be 0): {len(overlap)}")
    assert len(overlap) == 0, "Golden set leak detected!"
    print("  Golden set exclusion: OK")