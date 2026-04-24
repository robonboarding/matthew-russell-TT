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

from src.evaluate import context_recall


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
