"""
tests/test_parser.py -- Task 7: Unit tests for OutputParser
Run with:
    python -m pytest tests/test_parser.py -v
    OR without pytest:
    python tests/test_parser.py

Tests are grouped into 4 categories:
    1. TestCleaningSteps     -- each individual fix step
    2. TestFailureCases      -- inputs that should return success=False
    3. TestRealWorldOutputs  -- patterns seen in actual LLM responses
    4. TestParseResult       -- ParseResult dataclass behavior
"""

import sys
import unittest
from pathlib import Path

# Make sure src/ is on the path regardless of where test is run from
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from output_parser import OutputParser, ParseResult, parse_sql


# -------------------------------------------------------
# 1. Individual cleaning steps
# -------------------------------------------------------

class TestCleaningSteps(unittest.TestCase):
    """Test each of the 11 cleaning steps in isolation."""

    def setUp(self):
        self.parser = OutputParser()

    def test_strips_sql_markdown_fence(self):
        raw = "```sql\nSELECT name FROM users\n```"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("stripped_markdown_fence", result.issues)

    def test_strips_plain_markdown_fence(self):
        raw = "```\nSELECT name FROM users\n```"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")

    def test_strips_sql_label_prefix(self):
        raw = "SQL: SELECT name FROM users"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("stripped_leading_label", result.issues)

    def test_strips_query_label_prefix(self):
        raw = "Query: SELECT name FROM users"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")

    def test_strips_numbered_prefix(self):
        raw = "1. SELECT name FROM users"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")

    def test_strips_explanation_prefix(self):
        raw = "Here is the SQL query:\nSELECT name FROM users"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("stripped_prefix_explanation", result.issues)

    def test_strips_explanation_suffix(self):
        raw = "SELECT name FROM users\n\nThis query returns all user names."
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("stripped_suffix_explanation", result.issues)

    def test_removes_trailing_semicolon(self):
        raw = "SELECT name FROM users;"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("removed_trailing_semicolon", result.issues)

    def test_collapses_whitespace(self):
        raw = "SELECT   name   FROM   users"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")

    def test_collapses_newlines_inside_sql(self):
        raw = "SELECT name\nFROM users\nWHERE id = 1"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users WHERE id = 1")

    def test_uppercases_keywords(self):
        raw = "select name from users where id = 1"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users WHERE id = 1")
        self.assertTrue(any("uppercased" in i for i in result.issues))

    def test_normalizes_double_quotes(self):
        raw = 'SELECT name FROM users WHERE region = "West"'
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users WHERE region = 'West'")
        self.assertIn("normalized_quotes", result.issues)

    def test_takes_first_statement_only(self):
        raw = "SELECT name FROM users; SELECT id FROM orders"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT name FROM users")
        self.assertIn("took_first_statement_only", result.issues)


# -------------------------------------------------------
# 2. Failure cases
# -------------------------------------------------------

class TestFailureCases(unittest.TestCase):
    """Inputs that should return success=False with empty sql."""

    def setUp(self):
        self.parser = OutputParser()

    def test_empty_string(self):
        result = self.parser.parse("")
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")
        self.assertNotEqual(result.error, "")

    def test_whitespace_only(self):
        result = self.parser.parse("   \n\t  ")
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")

    def test_no_sql_at_all(self):
        result = self.parser.parse("I cannot generate SQL for this request.")
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")
        self.assertEqual(result.error, "No SELECT found after cleaning")

    def test_explanation_only_no_sql(self):
        result = self.parser.parse(
            "The question is ambiguous. Please provide more context."
        )
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")

    def test_sql_too_short(self):
        # After cleaning, only 1 word remains -- not valid SQL
        result = self.parser.parse("SELECT")
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")

    def test_failure_has_error_message(self):
        result = self.parser.parse("")
        self.assertIsInstance(result.error, str)
        self.assertGreater(len(result.error), 0)

    def test_failure_sql_is_always_empty_string(self):
        """Guarantee that failed parses never return None or raw text."""
        bad_inputs = [
            "",
            "not sql at all",
            "maybe SELECT but not really",
        ]
        for raw in bad_inputs:
            result = self.parser.parse(raw)
            if not result.success:
                self.assertEqual(result.sql, "",
                    f"Expected empty string for failed parse of: {raw!r}")


# -------------------------------------------------------
# 3. Real-world LLM output patterns
# -------------------------------------------------------

class TestRealWorldOutputs(unittest.TestCase):
    """
    Patterns taken directly from real LLM outputs seen in Week 1
    smoke test and common LLM response styles.
    """

    def setUp(self):
        self.parser = OutputParser()

    def test_week1_quote_style_difference(self):
        """Week 1 gold_01: model used single quotes, expected double quotes."""
        raw = "SELECT Name FROM people WHERE Nationality != 'Russia'"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertEqual(result.sql, "SELECT Name FROM people WHERE Nationality != 'Russia'")

    def test_week1_added_aliases(self):
        """Week 1 gold_03: model added AS aliases to aggregations."""
        raw = "SELECT MAX(Capacity) AS max_capacity, AVG(Average) AS average FROM stadium"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("SELECT", result.sql)
        self.assertIn("FROM", result.sql)

    def test_week1_inner_join_variation(self):
        """Week 1 gold_05: model used INNER JOIN instead of JOIN."""
        raw = (
            "SELECT T2.Birth_Date FROM poker_player AS T1 "
            "INNER JOIN people AS T2 ON T1.People_ID = T2.People_ID "
            "ORDER BY T1.Earnings ASC LIMIT 1"
        )
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("INNER JOIN", result.sql)
        self.assertIn("ORDER BY", result.sql)

    def test_multiline_with_indentation(self):
        """Common LLM formatting: indented multiline SQL."""
        raw = (
            "SELECT\n"
            "    c.name,\n"
            "    SUM(o.amount) AS total\n"
            "FROM customers c\n"
            "JOIN orders o ON c.id = o.customer_id\n"
            "GROUP BY c.name"
        )
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("SELECT", result.sql)
        self.assertIn("JOIN", result.sql)
        self.assertIn("GROUP BY", result.sql)

    def test_combined_fence_and_lowercase(self):
        """Fence + lowercase keywords -- very common pattern."""
        raw = "```sql\nselect count(*) from airports group by country\n```"
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("SELECT", result.sql)
        self.assertIn("COUNT", result.sql)
        self.assertIn("GROUP BY", result.sql)

    def test_response_with_preamble_and_explanation(self):
        """LLM adds context before and after SQL."""
        raw = (
            "Sure! Here is the SQL query to answer your question:\n\n"
            "```sql\n"
            "SELECT name FROM products ORDER BY price DESC LIMIT 1;\n"
            "```\n\n"
            "This query returns the most expensive product."
        )
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("SELECT", result.sql)
        self.assertIn("ORDER BY", result.sql)
        self.assertNotIn(";", result.sql)
        self.assertNotIn("```", result.sql)
        self.assertNotIn("Sure", result.sql)

    def test_subquery_preserved(self):
        """Make sure NOT IN subqueries aren't damaged by parsing."""
        raw = (
            "SELECT Aircraft FROM aircraft "
            "WHERE Aircraft_ID NOT IN (SELECT Winning_Aircraft FROM MATCH)"
        )
        result = self.parser.parse(raw)
        self.assertTrue(result.success)
        self.assertIn("NOT IN", result.sql)
        self.assertIn("SELECT", result.sql.count("SELECT") and result.sql)


# -------------------------------------------------------
# 4. ParseResult dataclass behavior
# -------------------------------------------------------

class TestParseResult(unittest.TestCase):
    """Test the ParseResult dataclass directly."""

    def test_success_result_has_sql(self):
        result = parse_sql("SELECT name FROM users")
        self.assertTrue(result.success)
        self.assertNotEqual(result.sql, "")
        self.assertEqual(result.error, "")

    def test_failure_result_has_error(self):
        result = parse_sql("not sql")
        self.assertFalse(result.success)
        self.assertEqual(result.sql, "")
        self.assertNotEqual(result.error, "")

    def test_issues_is_always_list(self):
        result = parse_sql("SELECT name FROM users")
        self.assertIsInstance(result.issues, list)

    def test_raw_is_preserved(self):
        raw = "```sql\nSELECT name FROM users\n```"
        result = parse_sql(raw)
        self.assertEqual(result.raw, raw)

    def test_str_representation(self):
        result = parse_sql("SELECT name FROM users")
        s = str(result)
        self.assertIn("OK", s)

    def test_convenience_function_matches_class(self):
        raw = "select name from users"
        result_fn    = parse_sql(raw)
        result_class = OutputParser().parse(raw)
        self.assertEqual(result_fn.sql, result_class.sql)
        self.assertEqual(result_fn.success, result_class.success)


# -------------------------------------------------------
# Runner
# -------------------------------------------------------

if __name__ == "__main__":
    # Run with verbose output showing each test name
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    # Add test classes in logical order
    for cls in [
        TestCleaningSteps,
        TestFailureCases,
        TestRealWorldOutputs,
        TestParseResult,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if any tests failed (useful for CI later)
    sys.exit(0 if result.wasSuccessful() else 1)