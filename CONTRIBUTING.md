# Contributing to PolicyMind-AI

Thank you for your interest in contributing. PolicyMind-AI is an open-source research platform, and contributions of all kinds — code, documentation, evaluation datasets, and bug reports — are welcome.

## Ground Rules

- Be respectful and constructive in all interactions (see [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md)).
- Open an issue before starting large features so we can discuss scope and approach.
- All submissions are covered by the [MIT License](LICENSE).

## Development Setup

```bash
git clone https://github.com/nazishatta/PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform
cd PolicyMind-AI-RAG-Powered-Document-Intelligence-Platform

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
python -m spacy download en_core_web_sm   # optional — regex fallback used if absent

cp .env.example .env               # Set LLM_PROVIDER=mock for offline dev
```

## Running Tests

```bash
pytest --cov=backend/app backend/tests/
```

## Linting and Type Checking

```bash
ruff check backend/
mypy backend/app/
```

## Submitting a Pull Request

1. Fork the repository and create a branch from `main`.
2. Write or update tests for your change.
3. Ensure `pytest` and `ruff check` pass with no errors.
4. Open a PR with a clear description of the problem and solution.
5. Link any related issues in the PR description.

## Reporting Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/` to file bug reports, feature requests, or research questions.

## Areas Most Needing Contributions

| Area | Notes |
|---|---|
| Evaluation datasets | Labeled Q&A pairs over public policy documents |
| Additional LLM providers | Cohere, Mistral, local Ollama |
| Graph schema extensions | New entity and relation types |
| Frontend demo | Streamlit or React interface |
| Language support | Non-English policy document handling |
