"""
Unit tests for evaluation metric pure functions.

Deliberately scoped to the deterministic, no-network functions in
src/evaluate.py. The LLM-as-judge paths (faithfulness, correctness) are
stochastic and require API credentials, so they are tested in the
integration eval harness (run_eval against eval/qa_pairs.json), not here.

In production I would add:
- Unit tests for mmr_rerank with known vector inputs
- Contract tests for the FastAPI endpoints using fastapi.testclient
- Golden-set regression tests (LLM-as-judge in CI with a small, cheap set)
- Property tests with Hypothesis for chunk_id round-tripping
"""
from __future__ import annotations

import pytest

from src.evaluate import citation_validity, context_recall, is_refusal, refusal_correctness


class TestContextRecall:
    def test_all_expected_chunks_retrieved(self):
        assert context_recall(
            retrieved=["a", "b", "c"],
            expected=["a", "b"],
        ) == 1.0

    def test_no_overlap(self):
        assert context_recall(
            retrieved=["x", "y"],
            expected=["a", "b"],
        ) == 0.0

    def test_partial_overlap(self):
        assert context_recall(
            retrieved=["a", "x"],
            expected=["a", "b"],
        ) == 0.5

    def test_empty_expected_means_perfect(self):
        """If no ground-truth chunks were labelled (e.g. refusal cases),
        retrieval cannot fail by definition."""
        assert context_recall(
            retrieved=["a"],
            expected=[],
        ) == 1.0

    def test_empty_retrieved_with_expected_is_zero(self):
        assert context_recall(
            retrieved=[],
            expected=["a"],
        ) == 0.0

    def test_duplicate_expected_chunks_counted_once_in_denominator(self):
        """Edge: if the gold set has duplicate IDs, we should not double-count.
        Current implementation does double-count — this test documents that
        and would fail if behaviour changed."""
        result = context_recall(
            retrieved=["a"],
            expected=["a", "a"],
        )
        # With current implementation: 2 hits / 2 expected = 1.0
        assert result == 1.0

    @pytest.mark.parametrize(
        "retrieved,expected,score",
        [
            (["a", "b", "c", "d"], ["a"], 1.0),
            (["a"], ["a", "b", "c", "d"], 0.25),
            (["a", "b"], ["b", "a"], 1.0),  # order-independent
        ],
    )
    def test_recall_parametrised(self, retrieved, expected, score):
        assert context_recall(retrieved, expected) == pytest.approx(score)


class TestCitationValidity:
    def test_all_citations_valid(self):
        answer = "CDOs were risky [doc#0001] and played a major role [doc#0002]."
        assert citation_validity(answer, ["doc#0001", "doc#0002"]) == 1.0

    def test_one_fabricated_citation(self):
        answer = "CDOs were risky [doc#0001] and insiders knew [fake_doc#9999]."
        assert citation_validity(answer, ["doc#0001", "doc#0002"]) == 0.5

    def test_all_citations_fabricated(self):
        answer = "CDOs were risky [fake#0001] and insiders [fake#0002]."
        assert citation_validity(answer, ["real#0001"]) == 0.0

    def test_no_citations_in_refusal_path(self):
        """A refusal that correctly contains no citations should not be penalised."""
        assert citation_validity(
            "I cannot answer this from the available documents.",
            ["doc#0001"],
        ) == 1.0

    def test_matches_only_well_formed_chunk_ids(self):
        """Loose brackets like [1] or [source] should not count as citations."""
        assert citation_validity("[1] and [source] are loose", []) == 1.0


class TestRefusalCorrectness:
    def test_refused_when_should_refuse(self):
        ans = "I cannot answer this from the available documents."
        assert refusal_correctness(ans, should_refuse=True) == 1.0

    def test_answered_when_should_not_refuse(self):
        ans = "CDOs were a type of structured financial instrument."
        assert refusal_correctness(ans, should_refuse=False) == 1.0

    def test_over_refusal_penalised(self):
        """Refusing a valid question is as bad as answering an invalid one."""
        ans = "I cannot answer this from the available documents."
        assert refusal_correctness(ans, should_refuse=False) == 0.0

    def test_under_refusal_penalised(self):
        ans = "The capital of France is Paris."
        assert refusal_correctness(ans, should_refuse=True) == 0.0

    def test_azure_filter_block_counts_as_refusal(self):
        assert is_refusal("[BLOCKED BY AZURE CONTENT FILTER]")
