"""
output_parser.py -- Task 2: Extract clean SQL from raw LLM response
Used by: eval_runner.py, smoke_test.py

Problems this parser solves (real cases from Week 1 + common LLM patterns):
    1.  Markdown fences       -- ```sql ... ``` or ``` ... ```
    2.  Explanation prefix    -- "Here is the SQL: SELECT ..."
    3.  Explanation suffix    -- "SELECT ... \n\nThis query does X"
    4.  Lowercase keywords    -- select, from, where -> SELECT, FROM, WHERE
    5.  Trailing semicolons   -- SELECT name FROM t;  -> no semicolon
    6.  Extra whitespace      -- multiple spaces/newlines collapsed
    7.  Double quotes         -- WHERE name = "Russia" -> 'Russia'
    8.  Numbered prefix       -- "1. SELECT ..." -> "SELECT ..."
    9.  Leading label         -- "SQL: SELECT ..." -> "SELECT ..."
    10. Multiple statements   -- takes only the first SELECT statement
    11. Newlines inside SQL   -- SELECT\nname\nFROM -> SELECT name FROM
"""

import re
from dataclasses import dataclass


# -------------------------------------------------------
# SQL Keywords for normalization
# -------------------------------------------------------

SQL_KEYWORDS = [
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS",
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS",
    "ON", "AS", "DISTINCT", "ORDER", "BY", "GROUP", "HAVING",
    "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT", "CASE",
    "WHEN", "THEN", "ELSE", "END", "NULL", "IS", "LIKE", "BETWEEN",
    "COUNT", "SUM", "AVG", "MAX", "MIN", "ASC", "DESC",
    "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
    "TABLE", "INTO", "VALUES", "SET", "PRIMARY", "KEY",
    "FOREIGN", "REFERENCES",
]


# -------------------------------------------------------
# ParseResult dataclass
# -------------------------------------------------------

@dataclass
class ParseResult:
    """
    Result of parsing a raw LLM output.

    Attributes:
        raw        : original unmodified LLM output
        sql        : cleaned SQL string (empty string if parsing failed)
        success    : True if a valid SQL was extracted
        issues     : list of issues found and fixed during parsing
        error      : error message if success=False
    """
    raw    : str
    sql    : str
    success: bool
    issues : list[str]
    error  : str = ""

    def __str__(self) -> str:
        status = "OK" if self.success else "FAIL"
        issues = f"  issues: {self.issues}" if self.issues else ""
        return f"[{status}] {self.sql[:80]}{issues}"


# -------------------------------------------------------
# OutputParser
# -------------------------------------------------------

class OutputParser:
    """
    Cleans raw LLM output and extracts a single normalized SQL query.

    Usage:
        parser = OutputParser()
        result = parser.parse("```sql\nSELECT name FROM users\n```")
        print(result.sql)  # SELECT name FROM users
    """

    def parse(self, raw: str) -> ParseResult:
        """
        Parse raw LLM output into a clean SQL string.

        Args:
            raw: raw string returned by the LLM

        Returns:
            ParseResult with cleaned SQL and metadata
        """
        if not raw or not raw.strip():
            return ParseResult(
                raw=raw, sql="", success=False,
                issues=[], error="Empty input"
            )

        text   = raw.strip()
        issues = []

        # -- Step 1: Strip markdown code fences --
        text, found = self._strip_markdown_fences(text)
        if found:
            issues.append("stripped_markdown_fence")

        # -- Step 2: Strip leading labels like "SQL:", "Query:", "1." --
        text, found = self._strip_leading_label(text)
        if found:
            issues.append("stripped_leading_label")

        # -- Step 3: Strip explanation text before SELECT --
        text, found = self._strip_prefix_text(text)
        if found:
            issues.append("stripped_prefix_explanation")

        # -- Step 4: Take only the first SQL statement --
        text, found = self._take_first_statement(text)
        if found:
            issues.append("took_first_statement_only")

        # -- Step 5: Strip explanation text after SQL ends --
        text, found = self._strip_suffix_text(text)
        if found:
            issues.append("stripped_suffix_explanation")

        # -- Step 6: Remove trailing semicolon --
        if text.rstrip().endswith(";"):
            text = text.rstrip().rstrip(";").rstrip()
            issues.append("removed_trailing_semicolon")

        # -- Step 7: Collapse whitespace and newlines --
        original = text
        text = re.sub(r'\s+', ' ', text).strip()
        if text != original:
            issues.append("collapsed_whitespace")

        # -- Step 8: Uppercase SQL keywords --
        text, count = self._uppercase_keywords(text)
        if count > 0:
            issues.append(f"uppercased_{count}_keywords")

        # -- Step 9: Normalize double quotes to single quotes --
        text, found = self._normalize_quotes(text)
        if found:
            issues.append("normalized_quotes")

        # -- Step 10: Validate result has SELECT --
        if not re.search(r'\bSELECT\b', text.upper()):
            return ParseResult(
                raw=raw, sql="", success=False,
                issues=issues,
                error="No SELECT found after cleaning"
            )

        # -- Step 11: Validate minimum length --
        if len(text.split()) < 3:
            return ParseResult(
                raw=raw, sql="", success=False,
                issues=issues,
                error="SQL too short after cleaning"
            )

        return ParseResult(raw=raw, sql=text, success=True, issues=issues)


    # -------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------

    def _strip_markdown_fences(self, text: str) -> tuple[str, bool]:
        """Remove ```sql ... ``` or ``` ... ``` blocks."""
        # ```sql\n...\n``` or ```SQL\n...\n```
        match = re.search(r'```(?:sql|SQL)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            return match.group(1).strip(), True
        # Single backtick inline: `SELECT ...`
        match = re.search(r'`([^`]+)`', text)
        if match and 'SELECT' in match.group(1).upper():
            return match.group(1).strip(), True
        return text, False

    def _strip_leading_label(self, text: str) -> tuple[str, bool]:
        """Remove labels like 'SQL:', 'Query:', 'Answer:', '1.' before SELECT."""
        patterns = [
            r'^(?:SQL|Query|Answer|Result|Output)\s*:\s*',  # "SQL: SELECT..."
            r'^\d+\.\s*',                                    # "1. SELECT..."
            r'^-\s*',                                        # "- SELECT..."
        ]
        for pattern in patterns:
            new = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
            if new != text:
                return new, True
        return text, False

    def _strip_prefix_text(self, text: str) -> tuple[str, bool]:
        """Remove explanation text that appears before the first SELECT."""
        upper = text.upper()
        # Find the first SELECT that isn't inside quotes
        idx = upper.find('SELECT')
        if idx > 0:
            # Make sure what's before isn't just whitespace
            prefix = text[:idx].strip()
            if prefix:
                return text[idx:].strip(), True
        return text, False

    def _take_first_statement(self, text: str) -> tuple[str, bool]:
        """
        If multiple SQL statements exist (separated by semicolons or
        double newlines), take only the first one.
        """
        # Split on semicolon followed by whitespace + SELECT
        parts = re.split(r';\s*(?=SELECT)', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[0].strip(), True
        # Split on double newline + SELECT
        parts = re.split(r'\n\n+(?=SELECT)', text, flags=re.IGNORECASE)
        if len(parts) > 1:
            return parts[0].strip(), True
        return text, False

    def _strip_suffix_text(self, text: str) -> tuple[str, bool]:
        """
        Remove explanation text that appears after the SQL ends.
        SQL ends at: double newline, 'This query', 'Note:', 'Explanation:'
        """
        # Common patterns that signal end of SQL
        stop_patterns = [
            r'\n\n+',                          # blank line
            r'\n(?:This|Note|The above|Here)', # explanation starts
            r'\n--\s',                         # SQL comment line
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, text)
            if match:
                candidate = text[:match.start()].strip()
                if 'SELECT' in candidate.upper():
                    return candidate, True
        return text, False

    def _uppercase_keywords(self, text: str) -> tuple[str, int]:
        """Uppercase all SQL keywords. Returns (new_text, count_changed)."""
        count = 0
        for kw in SQL_KEYWORDS:
            new = re.sub(rf'\b{kw}\b', kw, text, flags=re.IGNORECASE)
            if new != text:
                count += 1
                text = new
        return text, count

    def _normalize_quotes(self, text: str) -> tuple[str, bool]:
        """
        Replace double-quoted string literals with single quotes.
        Careful not to touch column/table names in double quotes.
        Only replaces WHERE col = "value" patterns.
        """
        # Match = "value" or != "value" or IN ("value") etc.
        pattern = r'(=\s*|!=\s*|<>\s*|IN\s*\(|,\s*)"([^"]*)"'
        new = re.sub(pattern, lambda m: m.group(1) + "'" + m.group(2) + "'", text)
        return new, new != text


# -------------------------------------------------------
# Convenience function
# -------------------------------------------------------

def parse_sql(raw: str) -> ParseResult:
    """Shorthand for OutputParser().parse(raw)."""
    return OutputParser().parse(raw)


# -------------------------------------------------------
# Smoke test / demo
# -------------------------------------------------------

if __name__ == "__main__":
    parser = OutputParser()

    test_cases = [
        # (description, raw_input, expected_clean_sql)
        (
            "markdown fence with sql tag",
            "```sql\nSELECT name FROM users WHERE region = 'West'\n```",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "markdown fence no tag",
            "```\nselect name from users\n```",
            "SELECT name FROM users",
        ),
        (
            "explanation prefix",
            "Here is the SQL query for your request:\nSELECT name FROM users WHERE region = 'West'",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "explanation suffix",
            "SELECT name FROM users WHERE region = 'West'\n\nThis query selects all users in the West region.",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "lowercase keywords",
            "select name from users where region = 'West'",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "trailing semicolon",
            "SELECT name FROM users WHERE region = 'West';",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "double quotes",
            'SELECT name FROM users WHERE region = "West"',
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "numbered prefix",
            "1. SELECT name FROM users WHERE region = 'West'",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "SQL label prefix",
            "SQL: SELECT name FROM users WHERE region = 'West'",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "newlines inside SQL",
            "SELECT name\nFROM users\nWHERE region = 'West'",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "combined mess",
            "```sql\n-- Here is the answer:\nselect name from users where region = \"West\";\n```\nThis returns all users.",
            "SELECT name FROM users WHERE region = 'West'",
        ),
        (
            "empty input",
            "",
            "",
        ),
        (
            "no SQL at all",
            "I cannot generate SQL for this request.",
            "",
        ),
    ]

    print("OutputParser smoke test")
    print("=" * 65)
    passed = 0
    failed = 0

    for desc, raw, expected in test_cases:
        result = parser.parse(raw)
        ok = result.sql == expected
        status = "[+] PASS" if ok else "[x] FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"\n{status}  {desc}")
        if not ok:
            print(f"       Expected : {expected}")
            print(f"       Got      : {result.sql}")
        if result.issues:
            print(f"       Fixed    : {result.issues}")

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} cases")
    print("=" * 65)