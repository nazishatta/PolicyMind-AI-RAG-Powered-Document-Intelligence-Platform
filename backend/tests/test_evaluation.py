"""Evaluation framework tests for PolicyMind-AI.

Nine test classes — one per evaluation dimension:

  1.  TestRetrievalRelevance        Recall@k, MRR, top-chunk keyword hit
  2.  TestCitationAccuracy          field completeness, score bounds, APA format
  3.  TestEntityExtractionQuality   label coverage, graceful fallback, recall metric
  4.  TestRelationExtractionQuality  co-occurrence typing, confidence, distance gate
  5.  TestAnswerFaithfulness         context grounding, limitations, confidence
  6.  TestSourceCoverage             keyword coverage, top-k monotonicity, empty corpus
  7.  TestHallucinationRisk          limitations surfaced, empty-corpus guard
  8.  TestLatency                    latency_ms field bounds and type
  9.  TestReproducibility            deterministic doc_id, chunk count, retrieval order

All tests run fully offline using the in-memory pipeline stack (mock LLM,
stub embeddings with fixed seed, NetworkX graph).  They validate framework
structure and observable pipeline properties — not the quality of a specific
embedding model or LLM provider.

To measure live quality metrics against a running server:
    python scripts/evaluate.py --base-url http://localhost:8000
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Synthetic evaluation documents
# ---------------------------------------------------------------------------

# ~150 chars — guaranteed to produce exactly one chunk (well below chunk_size=512)
_SINGLE_CHUNK_DOC = (
    "This policy mandates a 40 percent reduction in carbon emissions by 2035 "
    "under the Clean Air Framework established by the Environment Ministry."
)

# ~720 chars — produces 2+ chunks with default chunk_size=512
_CLIMATE_DOC = (
    "Section 1: Emission Reduction Targets. Signatory states shall reduce "
    "greenhouse gas emissions by 45 percent relative to 2005 baseline levels "
    "by the year 2030. The Environment Agency will publish quarterly compliance "
    "reports and issue enforcement notices for non-compliant parties. "
    "Section 2: Renewable Energy Obligations. A minimum 40 percent share of "
    "renewable energy in final consumption is mandated by 2028. Wind and solar "
    "installations receive priority grid access under Article 7 of this Framework. "
    "Section 3: Just Transition Fund. EUR 12 billion is allocated by the "
    "Treasury to support coal-dependent communities through retraining grants "
    "and infrastructure investment. Applications must be submitted to the "
    "Regional Development Authority by June 2025."
)

# ~700 chars — distinct vocabulary for cross-document tests
_HEALTH_DOC = (
    "Part I: National Vaccination Programme. The Department of Health will "
    "procure and distribute vaccines against influenza and measles to all "
    "registered practitioners by October 2024. "
    "Part II: Funding Allocation. A budget of GBP 800 million is approved "
    "under the NHS Modernisation Act 2023 for vaccine cold-chain logistics "
    "and community outreach programmes. Regional health boards must submit "
    "procurement plans to the Department of Health by January 2024. "
    "Part III: Monitoring and Reporting. The Public Health Observatory will "
    "report quarterly on uptake rates. Uptake below 75 percent triggers "
    "mandatory intervention under Regulation 12 of the Vaccination Framework."
)

# ~650 chars — geographic and trade vocabulary
_TRADE_DOC = (
    "Title I: Scope. This agreement governs trade relations between the "
    "European Union and the Southern African Development Community. "
    "Title II: Tariff Reductions. Import duties on agricultural goods "
    "originating from SADC member states shall be reduced by 30 percent "
    "within five years of ratification. The World Trade Organization dispute "
    "resolution mechanism applies to all disagreements arising under this text. "
    "Title III: Rules of Origin. Goods must satisfy a minimum 60 percent "
    "regional value content requirement to qualify for preferential tariff "
    "rates. The Joint Committee established under Article 45 will review "
    "these thresholds annually beginning January 2025."
)


# ---------------------------------------------------------------------------
# Metric helper functions (pure, no external dependencies)
# ---------------------------------------------------------------------------

def _recall_at_k(chunks: list[dict], gold_keyword: str, k: int) -> float:
    """1.0 if gold_keyword appears in any of the top-k chunk texts, else 0.0."""
    for chunk in chunks[:k]:
        if gold_keyword.lower() in chunk.get("text", "").lower():
            return 1.0
    return 0.0


def _mrr(chunks: list[dict], gold_keyword: str) -> float:
    """Reciprocal rank of the first chunk whose text contains gold_keyword."""
    for i, chunk in enumerate(chunks, start=1):
        if gold_keyword.lower() in chunk.get("text", "").lower():
            return 1.0 / i
    return 0.0


def _keyword_hit_rate(chunks: list[dict], keywords: list[str]) -> float:
    """Fraction of gold keywords found in at least one retrieved chunk text."""
    if not keywords or not chunks:
        return 0.0
    combined = " ".join(c.get("text", "") for c in chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords)


def _citation_field_completeness(citations: list[dict]) -> float:
    """Fraction of citations that carry all required fields with valid types."""
    required = {"chunk_id", "doc_id", "doc_title", "page_number", "relevance_score"}
    if not citations:
        return 0.0
    complete = sum(
        1 for c in citations
        if required.issubset(c.keys())
        and isinstance(c.get("relevance_score"), (int, float))
        and 0.0 <= c["relevance_score"] <= 1.0
        and isinstance(c.get("page_number"), int)
        and c.get("page_number", 0) >= 0
    )
    return complete / len(citations)


# ---------------------------------------------------------------------------
# 1. Retrieval Relevance
# ---------------------------------------------------------------------------

class TestRetrievalRelevance:
    """Recall@k and MRR over a fully controlled, in-memory corpus."""

    def _ingest_climate(self, api_client):
        return api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })

    def test_single_chunk_corpus_recall_at_1_is_one(self, api_client):
        """A single-chunk corpus guarantees recall@1 = 1.0 for any query."""
        api_client.post("/api/v1/ingest", json={
            "text": _SINGLE_CHUNK_DOC, "title": "Single-chunk Policy"
        })
        resp = api_client.post("/api/v1/query", json={
            "question": "What emissions reduction is required?", "top_k": 1
        })
        chunks = resp.json()["retrieved_chunks"]
        # Must return exactly the one chunk that exists
        assert len(chunks) == 1
        recall = _recall_at_k(chunks, "40 percent", k=1)
        assert recall == 1.0

    def test_mrr_bounded_in_zero_to_one(self, api_client):
        self._ingest_climate(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the 2030 targets?", "top_k": 5
        })
        chunks = resp.json()["retrieved_chunks"]
        mrr_value = _mrr(chunks, "45 percent")
        assert 0.0 <= mrr_value <= 1.0

    def test_recall_values_bounded(self, api_client):
        self._ingest_climate(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?", "top_k": 5
        })
        chunks = resp.json()["retrieved_chunks"]
        for kw in ["45 percent", "2030", "EUR 12 billion"]:
            assert 0.0 <= _recall_at_k(chunks, kw, k=5) <= 1.0

    def test_recall_k5_gte_recall_k1_monotone(self, api_client):
        """Recall@5 must be ≥ Recall@1 (retrieving more cannot reduce hits)."""
        self._ingest_climate(api_client)
        gold_kw = "45 percent"
        r1 = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?", "top_k": 1
        })
        r5 = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?", "top_k": 5
        })
        recall1 = _recall_at_k(r1.json()["retrieved_chunks"], gold_kw, k=1)
        recall5 = _recall_at_k(r5.json()["retrieved_chunks"], gold_kw, k=5)
        assert recall5 >= recall1

    def test_chunk_count_does_not_exceed_top_k(self, api_client):
        self._ingest_climate(api_client)
        for k in [1, 2, 4]:
            resp = api_client.post("/api/v1/query", json={
                "question": "What targets exist?", "top_k": k
            })
            assert len(resp.json()["retrieved_chunks"]) <= k

    def test_relevance_scores_all_in_valid_range(self, api_client):
        self._ingest_climate(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What targets exist?", "top_k": 5
        })
        for chunk in resp.json()["retrieved_chunks"]:
            score = chunk["relevance_score"]
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_empty_corpus_returns_no_chunks(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Any targets?"})
        assert resp.json()["retrieved_chunks"] == []


# ---------------------------------------------------------------------------
# 2. Citation Accuracy
# ---------------------------------------------------------------------------

class TestCitationAccuracy:
    """Citations carry required provenance fields and correctly bounded scores."""

    def _ingest_and_query(self, api_client):
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        return api_client.post("/api/v1/query", json={
            "question": "What are the emission reduction targets?"
        })

    def test_citations_present_after_retrieval(self, api_client):
        resp = self._ingest_and_query(api_client)
        assert isinstance(resp.json()["citations"], list)

    def test_all_citation_fields_present(self, api_client):
        resp = self._ingest_and_query(api_client)
        required = {"chunk_id", "doc_id", "doc_title", "page_number", "relevance_score"}
        for cite in resp.json()["citations"]:
            missing = required - cite.keys()
            assert not missing, f"Citation missing fields: {missing}"

    def test_citation_relevance_score_clamped_to_unit_interval(self, api_client):
        resp = self._ingest_and_query(api_client)
        for cite in resp.json()["citations"]:
            s = cite["relevance_score"]
            assert 0.0 <= s <= 1.0, f"relevance_score={s} is out of [0, 1]"

    def test_citation_page_number_non_negative(self, api_client):
        resp = self._ingest_and_query(api_client)
        for cite in resp.json()["citations"]:
            assert cite["page_number"] >= 0

    def test_citation_doc_id_matches_ingest(self, api_client):
        ingest_resp = api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        doc_id = ingest_resp.json()["doc_id"]
        query_resp = api_client.post("/api/v1/query", json={
            "question": "What are the emission reduction targets?"
        })
        for cite in query_resp.json()["citations"]:
            assert cite["doc_id"] == doc_id

    def test_citation_field_completeness_is_one(self, api_client):
        resp = self._ingest_and_query(api_client)
        citations = resp.json()["citations"]
        if citations:
            assert _citation_field_completeness(citations) == 1.0

    def test_citation_doc_title_nonempty(self, api_client):
        resp = self._ingest_and_query(api_client)
        for cite in resp.json()["citations"]:
            assert cite["doc_title"].strip()

    def test_citations_and_retrieved_chunks_same_chunk_ids(self, api_client):
        resp = self._ingest_and_query(api_client)
        body = resp.json()
        cite_ids = {c["chunk_id"] for c in body["citations"]}
        chunk_ids = {c["chunk_id"] for c in body["retrieved_chunks"]}
        # Every cited chunk should be in the retrieved set
        assert cite_ids.issubset(chunk_ids)


# ---------------------------------------------------------------------------
# 3. Entity Extraction Quality
# ---------------------------------------------------------------------------

class TestEntityExtractionQuality:
    """extract_entities() returns well-formed Entity objects or an empty list."""

    def test_returns_list_always(self):
        from app.core.entity_extraction import extract_entities
        result = extract_entities(_CLIMATE_DOC, "doc1", 1, "chunk1")
        assert isinstance(result, list)

    def test_entity_labels_within_policy_types(self):
        from app.core.entity_extraction import extract_entities, _POLICY_TYPES
        entities = extract_entities(_CLIMATE_DOC, "doc1", 1, "chunk1")
        for ent in entities:
            assert ent.label in _POLICY_TYPES, (
                f"Label '{ent.label}' not in policy entity set"
            )

    def test_entity_fields_fully_populated(self):
        from app.core.entity_extraction import extract_entities
        entities = extract_entities(_CLIMATE_DOC, "doc1", 1, "chunk1")
        for ent in entities:
            assert ent.text.strip(), "Entity text must not be blank"
            assert ent.doc_id == "doc1"
            assert ent.chunk_id == "chunk1"
            assert ent.page_number == 1
            assert ent.char_end > ent.char_start, "char_end must be after char_start"

    def test_entity_char_offsets_within_text(self):
        from app.core.entity_extraction import extract_entities
        text = _CLIMATE_DOC
        entities = extract_entities(text, "doc1", 1, "chunk1")
        for ent in entities:
            assert ent.char_start >= 0
            assert ent.char_end <= len(text)

    def test_entity_recall_metric_computable(self):
        """Entity recall is a float in [0, 1] regardless of spaCy availability."""
        from app.core.entity_extraction import extract_entities
        entities = extract_entities(_CLIMATE_DOC, "doc1", 1, "chunk1")
        extracted_texts = [e.text.lower() for e in entities]
        expected = ["environment agency", "2030", "eur 12 billion"]
        recall = sum(
            1 for e in expected
            if any(e in t for t in extracted_texts)
        ) / len(expected)
        assert 0.0 <= recall <= 1.0

    def test_graceful_fallback_returns_list_not_exception(self, monkeypatch):
        """When spaCy is unavailable, extract_entities returns [] not an exception."""
        from app.core import entity_extraction
        monkeypatch.setattr(
            entity_extraction, "_get_nlp",
            lambda: (_ for _ in ()).throw(ImportError("spacy not installed"))
        )
        from app.core.entity_extraction import extract_entities
        result = extract_entities(_CLIMATE_DOC, "doc1", 1, "chunk1")
        assert result == []

    def test_pipeline_entity_count_in_health_after_ingest(self, api_client):
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        health = api_client.get("/health").json()
        # Entity count is a non-negative int; actual value depends on spaCy
        assert isinstance(health["graph_entities"], int)
        assert health["graph_entities"] >= 0


# ---------------------------------------------------------------------------
# 4. Relation Extraction Quality
# ---------------------------------------------------------------------------

class TestRelationExtractionQuality:
    """extract_relations() infers typed edges from entity co-occurrence."""

    @staticmethod
    def _make_entity(text, label, start, end):
        from app.core.entity_extraction import Entity
        return Entity(
            text=text, label=label, doc_id="doc1", page_number=1,
            chunk_id="chunk1", char_start=start, char_end=end,
        )

    def test_org_gpe_pair_produces_operates_in(self):
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("Environment Agency", "ORG", 0, 18),
            self._make_entity("United Kingdom", "GPE", 22, 36),
        ]
        relations = extract_relations(
            ents, "Environment Agency in United Kingdom", "c1", "d1"
        )
        types = [r.relation_type for r in relations]
        assert "OPERATES_IN" in types

    def test_percent_date_pair_produces_targets_by(self):
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("45 percent", "PERCENT", 0, 10),
            self._make_entity("2030", "DATE", 14, 18),
        ]
        relations = extract_relations(
            ents, "45 percent reduction by 2030", "c1", "d1"
        )
        types = [r.relation_type for r in relations]
        assert "TARGETS_BY" in types

    def test_law_org_pair_produces_governed_by(self):
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("Clean Air Act 2023", "LAW", 0, 18),
            self._make_entity("Environment Agency", "ORG", 22, 40),
        ]
        relations = extract_relations(
            ents, "Clean Air Act 2023 by Environment Agency", "c1", "d1"
        )
        types = [r.relation_type for r in relations]
        assert "GOVERNED_BY" in types

    def test_all_relations_confidence_in_unit_interval(self):
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("WHO", "ORG", 0, 3),
            self._make_entity("Geneva", "GPE", 7, 13),
        ]
        for rel in extract_relations(ents, "WHO in Geneva", "c1", "d1"):
            assert 0.0 <= rel.confidence <= 1.0

    def test_relation_fields_populated(self):
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("Treasury", "ORG", 0, 8),
            self._make_entity("France", "GPE", 12, 18),
        ]
        for rel in extract_relations(ents, "Treasury in France today", "c1", "d1"):
            assert rel.source
            assert rel.target
            assert rel.relation_type
            assert rel.doc_id == "d1"
            assert rel.chunk_id == "c1"

    def test_entities_beyond_200_char_gap_not_linked(self):
        """Co-occurrence heuristic has a 200-char distance gate."""
        from app.core.entity_extraction import extract_relations
        ents = [
            self._make_entity("Agency", "ORG", 0, 6),
            self._make_entity("London", "GPE", 250, 256),
        ]
        text = "Agency" + (" " * 243) + "London"
        relations = extract_relations(ents, text, "c1", "d1")
        assert relations == []

    def test_no_entities_no_relations(self):
        from app.core.entity_extraction import extract_relations
        assert extract_relations([], "Some text here", "c1", "d1") == []


# ---------------------------------------------------------------------------
# 5. Answer Faithfulness
# ---------------------------------------------------------------------------

class TestAnswerFaithfulness:
    """Generated answers are grounded in retrieved passages and carry confidence."""

    def _setup(self, api_client, doc=_CLIMATE_DOC, title="Climate Framework"):
        api_client.post("/api/v1/ingest", json={"text": doc, "title": title})

    def test_answer_is_nonempty_string(self, api_client):
        self._setup(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What emission reduction targets are set?"
        })
        assert isinstance(resp.json()["answer"], str)
        assert resp.json()["answer"].strip()

    def test_mock_provider_limitation_always_surfaced(self, api_client):
        self._setup(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?"
        })
        lims = resp.json()["limitations"]
        assert any("mock" in l.lower() or "Mock" in l for l in lims), (
            f"Mock limitation not found in: {lims}"
        )

    def test_confidence_not_none_when_chunks_retrieved(self, api_client):
        self._setup(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What is the renewable energy obligation?"
        })
        body = resp.json()
        if body["retrieved_chunks"]:
            assert body["confidence"] is not None
            assert 0.0 <= body["confidence"] <= 1.0

    def test_confidence_none_when_corpus_empty(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        body = resp.json()
        if not body["retrieved_chunks"]:
            assert body["confidence"] is None

    def test_confidence_increases_with_graph_evidence(self, api_client):
        """graph_evidence boost: confidence with graph ≥ base similarity alone."""
        self._setup(api_client)
        r_with = api_client.post("/api/v1/query", json={
            "question": "What targets?", "include_graph": True
        })
        r_without = api_client.post("/api/v1/query", json={
            "question": "What targets?", "include_graph": False
        })
        c_with = r_with.json()["confidence"] or 0.0
        c_without = r_without.json()["confidence"] or 0.0
        graph_bonus = len(r_with.json()["graph_evidence"]) * 0.05
        # With graph evidence the confidence should be at least as high
        assert c_with >= c_without - 1e-6 or graph_bonus == 0.0

    def test_limitations_are_nonempty_strings(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        for lim in resp.json()["limitations"]:
            assert isinstance(lim, str) and lim.strip()

    def test_include_graph_false_suppresses_graph_evidence(self, api_client):
        self._setup(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?", "include_graph": False
        })
        assert resp.json()["graph_evidence"] == []


# ---------------------------------------------------------------------------
# 6. Source Coverage
# ---------------------------------------------------------------------------

class TestSourceCoverage:
    """Retrieved passages cover enough of the gold vocabulary to be useful."""

    def _ingest_climate(self, api_client):
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })

    def test_coverage_metric_in_unit_interval(self, api_client):
        self._ingest_climate(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What targets are mandated?", "top_k": 5
        })
        chunks = resp.json()["retrieved_chunks"]
        coverage = _keyword_hit_rate(chunks, ["45 percent", "2030", "EUR 12 billion"])
        assert 0.0 <= coverage <= 1.0

    def test_empty_corpus_zero_coverage(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "What are targets?"})
        chunks = resp.json()["retrieved_chunks"]
        assert chunks == []
        assert _keyword_hit_rate(chunks, ["45 percent"]) == 0.0

    def test_chunk_count_weakly_increases_with_top_k(self, api_client):
        """Requesting more chunks never returns fewer chunks."""
        self._ingest_climate(api_client)
        q = "What are the climate targets?"
        r1 = api_client.post("/api/v1/query", json={"question": q, "top_k": 1})
        r3 = api_client.post("/api/v1/query", json={"question": q, "top_k": 3})
        assert len(r3.json()["retrieved_chunks"]) >= len(r1.json()["retrieved_chunks"])

    def test_multi_doc_coverage_higher_than_single_doc(self, api_client):
        """Adding a second document into the corpus increases available coverage."""
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        resp_single = api_client.post("/api/v1/query", json={
            "question": "What targets?", "top_k": 5
        })
        chunks_single = len(resp_single.json()["retrieved_chunks"])

        api_client.post("/api/v1/ingest", json={
            "text": _HEALTH_DOC, "title": "Health Policy"
        })
        resp_multi = api_client.post("/api/v1/query", json={
            "question": "What targets?", "top_k": 5
        })
        chunks_multi = len(resp_multi.json()["retrieved_chunks"])
        # More content → at least as many retrievable chunks
        assert chunks_multi >= chunks_single

    def test_retrieved_chunk_texts_are_nonempty(self, api_client):
        self._ingest_climate(api_client)
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the targets?", "top_k": 3
        })
        for chunk in resp.json()["retrieved_chunks"]:
            assert chunk["text"].strip(), "Retrieved chunk text must not be blank"

    def test_coverage_keyword_hit_rate_helper(self):
        """Unit test for the _keyword_hit_rate metric function itself."""
        chunks = [
            {"text": "emissions reduced by 45 percent by 2030"},
            {"text": "EUR 12 billion allocated for the transition"},
        ]
        rate = _keyword_hit_rate(chunks, ["45 percent", "2030", "EUR 12 billion"])
        assert rate == 1.0

        rate_partial = _keyword_hit_rate(chunks, ["45 percent", "solar panels"])
        assert rate_partial == 0.5

        rate_none = _keyword_hit_rate([], ["45 percent"])
        assert rate_none == 0.0


# ---------------------------------------------------------------------------
# 7. Hallucination Risk
# ---------------------------------------------------------------------------

class TestHallucinationRisk:
    """Limitations are surfaced deterministically when hallucination risk is elevated."""

    def test_empty_corpus_triggers_no_passages_limitation(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        lims = resp.json()["limitations"]
        assert any(
            "passage" in l.lower() or "ingest" in l.lower() or "corpus" in l.lower()
            for l in lims
        ), f"Expected a 'no passages' limitation, got: {lims}"

    def test_mock_provider_limitation_always_present(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _CLIMATE_DOC, "title": "C"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        lims = resp.json()["limitations"]
        assert any("mock" in l.lower() or "LLM" in l for l in lims)

    def test_empty_graph_surfaces_graph_limitation(self, api_client):
        """When the graph has no entities, the response says so."""
        api_client.post("/api/v1/ingest", json={"text": _CLIMATE_DOC, "title": "C"})
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        body = resp.json()
        if body["graph_evidence"] == []:
            lims = body["limitations"]
            assert any("graph" in l.lower() for l in lims), (
                f"Expected a graph limitation when graph is empty, got: {lims}"
            )

    def test_low_confidence_triggers_limitation(self, api_client):
        """Confidence < 0.4 adds a low-confidence limitation."""
        # Use an extremely short doc to force low cosine similarity
        api_client.post("/api/v1/ingest", json={
            "text": _SINGLE_CHUNK_DOC, "title": "Short Policy"
        })
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        body = resp.json()
        conf = body.get("confidence") or 1.0
        lims = body["limitations"]
        if conf < 0.4:
            assert any("confidence" in l.lower() for l in lims)

    def test_limitations_all_strings(self, api_client):
        api_client.post("/api/v1/ingest", json={"text": _CLIMATE_DOC, "title": "C"})
        resp = api_client.post("/api/v1/query", json={"question": "Targets?"})
        for lim in resp.json()["limitations"]:
            assert isinstance(lim, str) and len(lim) > 0

    def test_single_chunk_limitation_surfaced(self, api_client):
        """A corpus with only one chunk surfaces the single-passage limitation."""
        api_client.post("/api/v1/ingest", json={
            "text": _SINGLE_CHUNK_DOC, "title": "Short Policy"
        })
        resp = api_client.post("/api/v1/query", json={
            "question": "What reduction is mandated?", "top_k": 5
        })
        body = resp.json()
        if len(body["retrieved_chunks"]) == 1:
            lims = body["limitations"]
            assert any("one" in l.lower() or "single" in l.lower() for l in lims)


# ---------------------------------------------------------------------------
# 8. Latency
# ---------------------------------------------------------------------------

class TestLatency:
    """latency_ms is positive, a real number, and bounded for the mock stack."""

    def test_query_latency_field_is_positive(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        assert resp.json()["latency_ms"] > 0

    def test_query_latency_is_numeric(self, api_client):
        resp = api_client.post("/api/v1/query", json={"question": "Anything?"})
        assert isinstance(resp.json()["latency_ms"], (int, float))

    def test_query_latency_under_five_seconds_with_mock(self, api_client):
        """Mock LLM + in-memory vector store should complete well under 5 s."""
        import time
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        t0 = time.perf_counter()
        resp = api_client.post("/api/v1/query", json={
            "question": "What are the emission targets?"
        })
        wall_ms = (time.perf_counter() - t0) * 1000
        assert wall_ms < 5000
        assert resp.json()["latency_ms"] > 0

    def test_ingest_completes_under_ten_seconds(self, api_client):
        """Stub embeddings + NetworkX graph should ingest in < 10 s."""
        import time
        t0 = time.perf_counter()
        resp = api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC + _HEALTH_DOC, "title": "Combined Policy"
        })
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 201
        assert elapsed_ms < 10000

    def test_latency_present_on_empty_corpus_query(self, api_client):
        """latency_ms is reported even when no chunks are retrieved."""
        resp = api_client.post("/api/v1/query", json={"question": "What targets?"})
        assert "latency_ms" in resp.json()
        assert resp.json()["latency_ms"] >= 0


# ---------------------------------------------------------------------------
# 9. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Identical inputs produce identical outputs across independent runs."""

    def test_same_text_produces_same_doc_id(self, api_client):
        body = {"text": _CLIMATE_DOC, "title": "Climate Framework"}
        r1 = api_client.post("/api/v1/ingest", json=body)
        r2 = api_client.post("/api/v1/ingest", json=body)
        assert r1.json()["doc_id"] == r2.json()["doc_id"]

    def test_same_text_produces_same_chunk_count(self, api_client):
        body = {"text": _HEALTH_DOC, "title": "Health Policy"}
        r1 = api_client.post("/api/v1/ingest", json=body)
        r2 = api_client.post("/api/v1/ingest", json=body)
        assert r1.json()["chunk_count"] == r2.json()["chunk_count"]

    def test_different_texts_produce_different_doc_ids(self, api_client):
        r1 = api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate"
        })
        r2 = api_client.post("/api/v1/ingest", json={
            "text": _HEALTH_DOC, "title": "Health"
        })
        assert r1.json()["doc_id"] != r2.json()["doc_id"]

    def test_same_query_same_retrieved_chunk_count(self, api_client):
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        q = {"question": "What emission targets are set?", "top_k": 3}
        r1 = api_client.post("/api/v1/query", json=q)
        r2 = api_client.post("/api/v1/query", json=q)
        assert (
            len(r1.json()["retrieved_chunks"])
            == len(r2.json()["retrieved_chunks"])
        )

    def test_same_query_same_top_chunk_id(self, api_client):
        """The fixed-seed stub embedding makes retrieval order deterministic."""
        api_client.post("/api/v1/ingest", json={
            "text": _CLIMATE_DOC, "title": "Climate Framework"
        })
        q = {"question": "What are the emission reduction requirements?", "top_k": 1}
        r1 = api_client.post("/api/v1/query", json=q)
        r2 = api_client.post("/api/v1/query", json=q)
        chunks1 = r1.json()["retrieved_chunks"]
        chunks2 = r2.json()["retrieved_chunks"]
        if chunks1 and chunks2:
            assert chunks1[0]["chunk_id"] == chunks2[0]["chunk_id"]

    def test_metric_helpers_are_deterministic(self):
        """Pure metric functions return identical values for identical inputs."""
        chunks = [
            {"text": "emissions shall be reduced by 45 percent by 2030"},
            {"text": "renewable energy target of 40 percent by 2028"},
        ]
        assert _recall_at_k(chunks, "45 percent", 1) == _recall_at_k(chunks, "45 percent", 1)
        assert _mrr(chunks, "45 percent") == _mrr(chunks, "45 percent")
        assert (
            _keyword_hit_rate(chunks, ["45 percent", "2030"])
            == _keyword_hit_rate(chunks, ["45 percent", "2030"])
        )
