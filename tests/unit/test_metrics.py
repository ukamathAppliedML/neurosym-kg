"""
Unit tests for evaluation metrics.
"""

import pytest
from neurosym_kg.evaluation.metrics import (
    normalize_answer,
    exact_match,
    f1_score,
    hits_at_k,
    mean_reciprocal_rank,
    MetricsCalculator,
)


class TestNormalizeAnswer:
    """Tests for answer normalization."""

    def test_lowercase(self):
        assert normalize_answer("PARIS") == "paris"

    def test_remove_articles(self):
        assert normalize_answer("the Eiffel Tower") == "eiffel tower"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_remove_punctuation(self):
        assert normalize_answer("Hello, World!") == "hello world"

    def test_normalize_whitespace(self):
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestExactMatch:
    """Tests for exact match metric."""

    def test_exact_match_true(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_exact_match_case_insensitive(self):
        assert exact_match("paris", "PARIS") == 1.0

    def test_exact_match_with_articles(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_exact_match_false(self):
        assert exact_match("Paris", "London") == 0.0

    def test_exact_match_multiple_answers(self):
        assert exact_match("Paris", ["Paris", "London"]) == 1.0
        assert exact_match("Berlin", ["Paris", "London"]) == 0.0


class TestF1Score:
    """Tests for F1 score metric."""

    def test_perfect_f1(self):
        assert f1_score("Paris France", "Paris France") == 1.0

    def test_partial_f1(self):
        # "Paris" appears in both, "France" only in prediction
        score = f1_score("Paris France", "Paris")
        assert 0 < score < 1

    def test_no_overlap_f1(self):
        assert f1_score("Paris", "London") == 0.0

    def test_f1_with_multiple_answers(self):
        # Should return best F1 across all answers
        score = f1_score("Paris France", ["Paris", "Berlin"])
        assert score > 0


class TestHitsAtK:
    """Tests for Hits@K metric."""

    def test_hits_at_1_found(self):
        predictions = ["Paris", "London", "Berlin"]
        assert hits_at_k(predictions, "Paris", k=1) == 1.0

    def test_hits_at_1_not_found(self):
        predictions = ["London", "Paris", "Berlin"]
        assert hits_at_k(predictions, "Paris", k=1) == 0.0

    def test_hits_at_3_found(self):
        predictions = ["London", "Berlin", "Paris"]
        assert hits_at_k(predictions, "Paris", k=3) == 1.0

    def test_hits_at_k_multiple_answers(self):
        predictions = ["London", "Paris", "Berlin"]
        assert hits_at_k(predictions, ["Paris", "Rome"], k=3) == 1.0


class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_mrr_first_position(self):
        predictions = ["Paris", "London", "Berlin"]
        assert mean_reciprocal_rank(predictions, "Paris") == 1.0

    def test_mrr_second_position(self):
        predictions = ["London", "Paris", "Berlin"]
        assert mean_reciprocal_rank(predictions, "Paris") == 0.5

    def test_mrr_third_position(self):
        predictions = ["London", "Berlin", "Paris"]
        assert mean_reciprocal_rank(predictions, "Paris") == pytest.approx(1/3)

    def test_mrr_not_found(self):
        predictions = ["London", "Berlin", "Rome"]
        assert mean_reciprocal_rank(predictions, "Paris") == 0.0


class TestMetricsCalculator:
    """Tests for MetricsCalculator class."""

    def test_single_prediction(self):
        calc = MetricsCalculator(dataset_name="test")
        calc.add_prediction(
            prediction="Paris",
            ground_truth="Paris",
            question="Capital of France?"
        )

        result = calc.compute()
        assert result.accuracy == 1.0
        assert result.f1 == 1.0

    def test_multiple_predictions(self):
        calc = MetricsCalculator()
        calc.add_prediction("Paris", "Paris")
        calc.add_prediction("Berlin", "Paris")  # Wrong

        result = calc.compute()
        assert result.accuracy == 0.5

    def test_empty_calculator(self):
        calc = MetricsCalculator()
        result = calc.compute()

        assert result.accuracy == 0.0
        assert len(result.predictions) == 0

    def test_reset(self):
        calc = MetricsCalculator()
        calc.add_prediction("Paris", "Paris")
        calc.reset()

        assert len(calc.predictions) == 0

    def test_with_metadata(self):
        calc = MetricsCalculator()
        calc.add_prediction(
            prediction="Paris",
            ground_truth="Paris",
            question="Q1",
            confidence=0.9,
            latency_ms=100.0,
            custom_field="value"
        )

        result = calc.compute()
        assert result.predictions[0].confidence == 0.9
        assert result.predictions[0].latency_ms == 100.0
        assert result.predictions[0].metadata["custom_field"] == "value"
