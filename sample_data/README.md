# Sample Data

> **Important:** This directory contains only lightweight demonstration content.
> No real policy documents, copyrighted materials, or large datasets are stored here.

---

## Contents

| File | Description |
|---|---|
| `sample_policy.txt` | Synthetic 1 200-word policy document created for software testing and demos. Not a real policy. Do not cite. |

---

## Using Real Policy Documents

PolicyMind-AI is designed for URL-based or API-based document access. Users supply their own document sources. No real documents are bundled with this repository.

### Recommended public sources

| Source | URL | License |
|---|---|---|
| EUR-Lex (EU legislation) | eur-lex.europa.eu | Public domain (CC BY 4.0 where marked) |
| World Bank Open Knowledge | openknowledge.worldbank.org | Creative Commons (CC BY 3.0/4.0) |
| UN Document System | documents.un.org | Verify per document |
| UNDP Policy Documents | undp.org/publications | Verify per document |
| UK Government Publications | gov.uk/government/publications | Open Government Licence |
| US Federal Register | federalregister.gov | US Government Works (public domain) |
| OECD iLibrary | oecd-ilibrary.org | Verify per document; most require subscription |

### Ingesting a public document via URL

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:52021PC0557",
    "title": "Fit for 55 Package",
    "source_label": "EU Commission"
  }'
```

### Ingesting the local sample file

The sample file can be ingested as a text payload for local testing:

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "'"$(cat sample_data/sample_policy.txt)"'",
    "title": "Sample Policy (Demo)",
    "source_label": "PolicyMind-AI Demo"
  }'
```

Or using the Python script:

```bash
python scripts/ingest_sample.py
```

---

## Licensing and Compliance

- Always verify the license of the source document before ingestion.
- PolicyMind-AI does not cache or redistribute document content.
- Only chunk embeddings and extracted metadata are stored in the vector store.
- When using LLM providers (Anthropic, OpenAI), document excerpts are transmitted to those providers' APIs subject to their data handling policies.
- For sensitive or embargoed documents, use `LLM_PROVIDER=mock` and avoid committing document text to version control.

---

## Adding Your Own Sample Documents

To add a sample document for testing:

1. Confirm the document is in the public domain or licensed for unrestricted research use.
2. Place a lightweight plain-text excerpt (≤ 50 KB) in this directory.
3. Add an entry to this README documenting the source, URL, and license.
4. Do not commit full PDF files — use URL-based ingestion instead.
