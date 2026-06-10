# Sample Queries

Representative questions for testing and demonstrating PolicyMind-AI across common policy research scenarios. Each question is designed to exercise retrieval, entity recognition, and citation grounding against a policy document corpus.

---

## What to Look For in Responses

Every query response includes trust-layer fields alongside the answer:

| Field | What to check |
|---|---|
| `answer_type` | `cited` = well-grounded; `partial` = weak evidence; `refused` = context insufficient; `no_corpus` = nothing indexed |
| `evidence_quality` | `strong` (conf ≥ 0.7, ≥ 3 passages) / `moderate` / `weak` / `insufficient` |
| `confidence_note` | Explains the numeric score in plain English |
| `citations` | Verify each claim maps to a real `page_number` and `doc_title` |
| `limitations` | Always read these — they are deterministic disclosures, not LLM-generated hedging |

When using the mock LLM (`LLM_PROVIDER=mock`), `answer` will be a deterministic stub. Set `LLM_PROVIDER=anthropic` or `openai` for synthesised natural-language answers.

---

## Climate and Environment

1. What greenhouse gas emission reduction targets are specified, and by what year?
2. Which sectors are explicitly named as subject to binding emissions targets?
3. What financial mechanisms are established to support the transition to net zero?
4. How are non-compliant member states penalised under the regulation?
5. What is the minimum required share of renewable energy in final energy consumption?
6. How are land use and forestry obligations defined?
7. What conditions must a state meet to qualify for phased compliance?

## Economic Development and Just Transition

8. What budget allocations are made to low-income or fossil-fuel-dependent regions?
9. Which international financial institutions are named as implementation partners?
10. What conditions are attached to structural adjustment or fund disbursement provisions?
11. How are small and medium enterprises addressed in the policy?
12. What worker retraining or social protection measures are included?

## Governance and Accountability

13. Which bodies are responsible for monitoring and compliance?
14. What reporting obligations are placed on national governments?
15. How are civil society organisations engaged in the oversight process?
16. What mechanisms exist for citizens to challenge non-compliance?
17. What happens when a member state fails two consecutive reporting periods?

## Health and Social Policy

18. What universal health coverage commitments are made?
19. How are vulnerable populations defined and protected?
20. What targets are set for reducing maternal or child mortality?
21. How does the policy address mental health or disability inclusion?

## Education and Digital Access

22. What commitments are made to digital infrastructure in underserved communities?
23. What teacher training targets are specified?
24. How are language and cultural barriers addressed in education policy?

## Definitions and Legal Scope

25. How is "net zero" defined in the document?
26. What is the scope of application — which states or actors are bound?
27. When does the strategy enter into force?
28. What review mechanisms are specified, and how often do they occur?

---

## How to Run These Queries

**Via the API (curl):**

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What greenhouse gas emission reduction targets are specified?"}' \
  | python -m json.tool
```

**Scoped to a single document:**

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How are non-compliant states penalised?",
    "doc_id": "REPLACE_WITH_DOC_ID",
    "top_k": 3
  }' | python -m json.tool
```

**Without graph evidence (faster, vector-only):**

```bash
curl -s -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What fund supports the just transition?",
    "include_graph_evidence": false
  }' | python -m json.tool
```

**Via the smoke test:**

```bash
python scripts/smoke_test.py
```

**Via the evaluation script (against a live server):**

```bash
python scripts/evaluate.py --base-url http://localhost:8000 --report eval_report.txt
```

---

## Queries That Will Trigger Refusal

These questions are intentionally unanswerable from a climate policy corpus — they demonstrate the trust gate:

- "What is the GDP of France?"
- "Who won the 2022 FIFA World Cup?"
- "What are the side effects of ibuprofen?"

Expected response: `answer_type = "no_corpus"` (if corpus is empty) or `answer_type = "refused"` (if the LLM determines context is insufficient). The pipeline never fabricates answers for out-of-scope questions.
