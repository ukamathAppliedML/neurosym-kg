"""
Evaluation metrics for neuro-symbolic reasoning.

Provides standard metrics for evaluating QA performance on KG benchmarks.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    # Lowercase
    answer = answer.lower()
    # Remove articles
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Remove punctuation
    answer = re.sub(r"[^\w\s]", "", answer)
    # Normalize whitespace
    answer = " ".join(answer.split())
    return answer.strip()


def exact_match(prediction: str, ground_truth: str | list[str]) -> float:
    """
    Compute exact match score.

    Returns 1.0 if prediction matches any ground truth, 0.0 otherwise.
    """
    pred_norm = normalize_answer(prediction)

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    for gt in ground_truth:
        if pred_norm == normalize_answer(gt):
            return 1.0

    return 0.0


def f1_score(prediction: str, ground_truth: str | list[str]) -> float:
    """
    Compute token-level F1 score.

    Measures overlap between prediction and ground truth tokens.
    """
    pred_tokens = normalize_answer(prediction).split()

    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    best_f1 = 0.0
    for gt in ground_truth:
        gt_tokens = normalize_answer(gt).split()

        if not pred_tokens or not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        best_f1 = max(best_f1, f1)

    return best_f1


def hits_at_k(
    predictions: list[str],
    ground_truth: str | list[str],
    k: int = 1,
) -> float:
    """
    Compute Hits@K score.

    Returns 1.0 if any ground truth appears in top-K predictions.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    gt_normalized = {normalize_answer(gt) for gt in ground_truth}

    for pred in predictions[:k]:
        if normalize_answer(pred) in gt_normalized:
            return 1.0

    return 0.0


def mean_reciprocal_rank(
    predictions: list[str],
    ground_truth: str | list[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Returns 1/rank where rank is position of first correct answer.
    """
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]

    gt_normalized = {normalize_answer(gt) for gt in ground_truth}

    for i, pred in enumerate(predictions):
        if normalize_answer(pred) in gt_normalized:
            return 1.0 / (i + 1)

    return 0.0


@dataclass
class SinglePrediction:
    """A single prediction result."""

    question: str
    prediction: str
    ground_truth: list[str]
    reasoning_paths: list[str] = field(default_factory=list)
    confidence: float = 1.0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exact_match(self) -> float:
        return exact_match(self.prediction, self.ground_truth)

    @property
    def f1(self) -> float:
        return f1_score(self.prediction, self.ground_truth)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""

    predictions: list[SinglePrediction]
    dataset_name: str = ""

    # Aggregated metrics (computed lazily)
    _metrics: dict[str, float] | None = field(default=None, repr=False)

    @property
    def metrics(self) -> dict[str, float]:
        """Compute and return all metrics."""
        if self._metrics is None:
            self._compute_metrics()
        return self._metrics  # type: ignore

    def _compute_metrics(self) -> None:
        """Compute all aggregated metrics."""
        n = len(self.predictions)
        if n == 0:
            self._metrics = {
                "accuracy": 0.0,
                "f1": 0.0,
                "hits_at_1": 0.0,
                "hits_at_3": 0.0,
                "hits_at_5": 0.0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0,
            }
            return

        # Exact match / accuracy
        accuracy = sum(p.exact_match for p in self.predictions) / n

        # F1
        avg_f1 = sum(p.f1 for p in self.predictions) / n

        # Latency
        avg_latency = sum(p.latency_ms for p in self.predictions) / n

        # Confidence
        avg_confidence = sum(p.confidence for p in self.predictions) / n

        self._metrics = {
            "accuracy": accuracy,
            "f1": avg_f1,
            "hits_at_1": accuracy,  # Same as exact match for single predictions
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "num_samples": float(n),
        }

    @property
    def accuracy(self) -> float:
        return self.metrics["accuracy"]

    @property
    def f1(self) -> float:
        return self.metrics["f1"]

    @property
    def hits_at_1(self) -> float:
        return self.metrics["hits_at_1"]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset_name,
            "num_samples": len(self.predictions),
            "metrics": self.metrics,
        }

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            f"Evaluation Results: {self.dataset_name}",
            f"  Samples: {len(self.predictions)}",
            f"  Accuracy: {self.accuracy:.2%}",
            f"  F1 Score: {self.f1:.2%}",
            f"  Avg Latency: {self.metrics['avg_latency_ms']:.0f}ms",
            f"  Avg Confidence: {self.metrics['avg_confidence']:.2%}",
        ]
        return "\n".join(lines)

    def error_analysis(self, top_n: int = 10) -> list[SinglePrediction]:
        """Get top-N incorrect predictions for analysis."""
        incorrect = [p for p in self.predictions if p.exact_match == 0]
        # Sort by confidence (high confidence errors are more interesting)
        incorrect.sort(key=lambda p: -p.confidence)
        return incorrect[:top_n]


class MetricsCalculator:
    """
    Utility class for computing various QA metrics.

    Example:
        >>> calc = MetricsCalculator()
        >>> calc.add_prediction("Paris", ["Paris"], question="Capital of France?")
        >>> calc.add_prediction("Berlin", ["Paris"], question="Capital of France?")
        >>> result = calc.compute()
        >>> print(result.accuracy)  # 0.5
    """

    def __init__(self, dataset_name: str = "") -> None:
        self.dataset_name = dataset_name
        self.predictions: list[SinglePrediction] = []

    def add_prediction(
        self,
        prediction: str,
        ground_truth: str | list[str],
        question: str = "",
        reasoning_paths: list[str] | None = None,
        confidence: float = 1.0,
        latency_ms: float = 0.0,
        **metadata: Any,
    ) -> None:
        """Add a single prediction."""
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]

        self.predictions.append(
            SinglePrediction(
                question=question,
                prediction=prediction,
                ground_truth=ground_truth,
                reasoning_paths=reasoning_paths or [],
                confidence=confidence,
                latency_ms=latency_ms,
                metadata=metadata,
            )
        )

    def compute(self) -> EvaluationResult:
        """Compute aggregated metrics."""
        return EvaluationResult(
            predictions=self.predictions.copy(),
            dataset_name=self.dataset_name,
        )

    def reset(self) -> None:
        """Clear all predictions."""
        self.predictions.clear()
