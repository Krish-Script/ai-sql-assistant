# Contributing to AI SQL Assistant

Thank you for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/ai-sql-assistant`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`
5. Build embeddings: `python src/embed_pairs.py`
6. Run tests: `pytest tests/ -v`

## Development Guidelines

- All new features must include tests
- Run `pytest tests/ -v` before submitting a PR -- all 82 tests must pass
- SQL normalization changes must not break `tests/test_parser.py`
- API changes must be reflected in `docs/api_reference.md`

## Adding New NL->SQL Pairs

1. Add pairs to `data/raw/` in the appropriate format
2. Run `python src/collect.py` to resample
3. Run `python src/clean.py` to normalize
4. Run `python src/validator.py` to validate
5. Run `python src/embed_pairs.py` to rebuild the embeddings index

## Reporting Issues

Please include:
- Python version
- Error message and full traceback
- Steps to reproduce
- Expected vs actual behavior