# Interview Preparation — AI SQL Code Assistant

**5 Q&A pairs for technical interviews and portfolio conversations.**

---

**Q1: What problem does your project solve?**

Non-technical users can't write SQL even when they know exactly what
data they need. My system takes a natural language query and returns
valid SQL using retrieval-augmented generation — no SQL knowledge
required. It exposes a REST API and Streamlit UI so it can be integrated
into existing data workflows without any SQL prerequisite.

---

**Q2: How does the RAG component work?**

I embed all 64 training pairs using sentence-transformers
(all-MiniLM-L6-v2, 384-dim). At query time I encode the input query,
compute cosine similarity against the stored L2-normalized embeddings,
and inject the top-3 most similar NL->SQL pairs as few-shot examples
into the LLM prompt. This gives the model concrete SQL structure and
schema patterns to follow, rather than generating from scratch.

I also use an `exclude_ids` parameter to prevent the retriever from
returning the query's own pair when it's already in the index —
avoiding data leakage during evaluation.

---

**Q3: Why did RAG improve results so dramatically (+55pp)?**

Two reasons. First, the retrieved examples provide schema context and
SQL pattern reference that the model wouldn't have zero-shot — it sees
which tables and columns are relevant. Second, the model sees the
exact output format expected (clean SQL, no markdown, no explanation),
which reduces style variation and format violations. The improvement
is larger on the golden set (60%) than the full 64-pair set (67%
paradoxically higher) because retrieval quality varies — well-covered
query types benefit more than edge cases.

---

**Q4: What's the biggest limitation of your system?**

Retrieval quality is bounded by dataset coverage. If a user asks about
a query pattern not represented in the 64 pairs, the retriever returns
low-similarity results (sim < 0.3) that may actively mislead the model.
The fix is a larger, more diverse dataset — ideally 500+ pairs with
coverage across more domains and schema complexities. A similarity
threshold filter (e.g. only inject examples with sim > 0.5) would also
prevent low-quality retrieved examples from degrading performance.

---

**Q5: How did you evaluate it, and what would you improve about your evaluation?**

Three metrics: exact match via sqlglot AST normalization (structural
comparison), semantic similarity via sentence-transformers cosine
similarity (logical equivalence), and structural validity check via
sqlglot parsing. I maintained a hand-labeled golden set of 10 pairs
throughout for controlled comparisons across weeks.

What I'd improve: I'd add execution-based evaluation — actually running
the generated SQL against a test database and comparing result sets.
This would catch cases where the SQL is structurally valid but
logically wrong (e.g. wrong JOIN condition). Execution match is the
gold standard for NL-to-SQL evaluation and would give a more honest
picture of real-world correctness than exact match or semantic similarity.