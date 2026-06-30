# PolicyMind AI -- Demo Guide

A curated set of public policy documents and sample questions to test
PolicyMind AI without needing to find your own documents.

> All sample documents are publicly available under open government or
> Creative Commons licenses. PolicyMind AI does not store or redistribute
> document content -- only chunk embeddings are persisted locally.

---

## Quick Start -- Paste These URLs Directly Into the App

### World Bank Reports

**World Development Report 2024: The Middle Income Trap**
- URL: `https://documents1.worldbank.org/curated/en/099080824150598470/pdf/P1807451cdbcc60881afdc19c40acb2e017.pdf`
- Pages: 43
- Best for: economic policy, country comparisons, development strategy

**World Bank Climate Change Action Plan**
- URL: `https://documents1.worldbank.org/curated/en/251001468348196768/pdf/103431-REVISED-PUBLIC-WBGClimateChangeActionPlan.pdf`
- Pages: ~60
- Best for: climate policy, organization relationships, implementation

### WHO and Health Policy Reports

**WHO Mental Health Action Plan 2013-2030**
- URL: `https://iris.who.int/bitstream/handle/10665/345301/9789240031029-eng.pdf`
- Best for: mental health policy, implementation recommendations, country analysis

**WHO Global Tuberculosis Report 2023**
- URL: `https://iris.who.int/bitstream/handle/10665/373828/9789240083851-eng.pdf`
- Best for: disease burden statistics, public health interventions, global policy

**NHS Long Term Plan (UK)**
- URL: `https://www.longtermplan.nhs.uk/wp-content/uploads/2019/08/nhs-long-term-plan-version-1.2.pdf`
- Best for: healthcare system policy, implementation strategy, funding questions

### UN Reports

**UN Human Development Report 2023-24**
- URL: `https://hdr.undp.org/system/files/documents/global-report-document/hdr2023-24reporten.pdf`
- Pages: 100+
- Best for: human development, country comparisons, social policy

---

## Sample Questions by RAG Strategy

### Standard RAG -- Specific Factual Questions

    What are the main policy recommendations in this document?
    What evidence does the report provide to support its conclusions?
    What countries or regions are discussed in this report?
    What are the key challenges identified by the authors?
    What data or statistics are cited in this report?
    What populations are most affected according to this report?

### Map-Reduce -- Summarization Questions

Keywords that trigger this: summarize, overview, key findings, main themes

    Summarize the key findings of this document
    Give me an overview of the main themes in this report
    What are the most important points across this document?
    Provide a comprehensive summary of the policy recommendations
    What are the key takeaways from this report?

### GraphRAG -- Entity and Relationship Questions

Keywords that trigger this: relationship, organization, between, responsible for

    What is the relationship between economic growth and policy reform?
    Which organizations are responsible for implementing these policies?
    What is the relationship between poverty and public health outcomes?
    Which countries have the highest disease burden according to this report?
    What is the connection between mental health and economic development?

---

## Expected Output Quality

| Query Type | Sources Found | Confidence | Answer Type |
|---|---|---|---|
| Specific factual | 3-5 | Strong 65-75% | Document Answer |
| Summarization | 15-20 | Strong 65-75% | Single Document Summary |
| Entity/relationship | 5-10 | Moderate 50-70% | GraphRAG Answer |

---

## Tips for Best Results

- Use longer documents (20+ pages) for better retrieval diversity
- Be specific in questions for higher confidence scores
- Toggle GraphRAG on in Step 2 before building the knowledge base
- Upload multiple PDFs to test cross-document Map-Reduce comparison
- Click View Sources to see exact page citations for every answer

---

## Live Demo

Try the deployed app directly -- no installation needed:

**[Open PolicyMind AI on Hugging Face](https://nazishatta-policymind-ai.hf.space/)**

---

## Licensing

All documents listed above are published under open licenses:
- WHO: Creative Commons CC BY-NC-SA 3.0 IGO
- World Bank: Creative Commons CC BY 3.0
- NHS: Open Government Licence
- UN: Public domain for non-commercial use

PolicyMind AI does not cache, store, or redistribute document content.
Only chunk embeddings and extracted metadata are stored locally.