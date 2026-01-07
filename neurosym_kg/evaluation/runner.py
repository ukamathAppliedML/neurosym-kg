"""
Evaluation runner for benchmarking reasoners.

Provides tools for running evaluations and generating reports.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from neurosym_kg.core.interfaces import Reasoner
from neurosym_kg.evaluation.benchmarks import BaseBenchmark, BenchmarkExample
from neurosym_kg.evaluation.metrics import (
    EvaluationResult,
    MetricsCalculator,
    SinglePrediction,
)


@dataclass
class RunConfig:
    """Configuration for an evaluation run."""

    subset_size: int | None = None  # If set, only evaluate this many examples
    random_seed: int = 42  # For reproducible subsampling
    max_retries: int = 1  # Retries on reasoner failure
    timeout_seconds: float = 120.0  # Per-question timeout
    save_predictions: bool = True  # Save individual predictions
    verbose: bool = False


@dataclass
class RunReport:
    """Report from an evaluation run."""

    benchmark_name: str
    reasoner_name: str
    results: EvaluationResult
    config: RunConfig
    start_time: datetime
    end_time: datetime
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark": self.benchmark_name,
            "reasoner": self.reasoner_name,
            "metrics": self.results.metrics,
            "num_samples": len(self.results.predictions),
            "num_errors": len(self.errors),
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "config": {
                "subset_size": self.config.subset_size,
                "random_seed": self.config.random_seed,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save report to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        # Add predictions if configured
        if self.config.save_predictions:
            data["predictions"] = [
                {
                    "question": p.question,
                    "prediction": p.prediction,
                    "ground_truth": p.ground_truth,
                    "exact_match": p.exact_match,
                    "f1": p.f1,
                    "confidence": p.confidence,
                    "latency_ms": p.latency_ms,
                }
                for p in self.results.predictions
            ]

        # Add errors
        data["errors"] = self.errors

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=" * 60,
            f"Evaluation Report",
            f"=" * 60,
            f"Benchmark: {self.benchmark_name}",
            f"Reasoner: {self.reasoner_name}",
            f"Duration: {self.duration_seconds:.1f}s",
            f"",
            self.results.summary(),
            f"",
            f"Errors: {len(self.errors)}",
            f"=" * 60,
        ]
        return "\n".join(lines)


class BenchmarkRunner:
    """
    Runs evaluations of reasoners on benchmarks.

    Example:
        >>> runner = BenchmarkRunner(reasoner)
        >>> report = runner.evaluate(WebQSP(), config=RunConfig(subset_size=100))
        >>> print(report.summary())
    """

    def __init__(
        self,
        reasoner: Reasoner,
        pre_process: Callable[[BenchmarkExample], str] | None = None,
        post_process: Callable[[str], str] | None = None,
    ) -> None:
        """
        Initialize the runner.

        Args:
            reasoner: The reasoner to evaluate
            pre_process: Optional function to preprocess questions
            post_process: Optional function to postprocess answers
        """
        self.reasoner = reasoner
        self.pre_process = pre_process or (lambda ex: ex.question)
        self.post_process = post_process or (lambda x: x)

    def evaluate(
        self,
        benchmark: BaseBenchmark,
        config: RunConfig | None = None,
    ) -> RunReport:
        """
        Evaluate the reasoner on a benchmark.

        Args:
            benchmark: The benchmark to evaluate on
            config: Evaluation configuration

        Returns:
            Evaluation report
        """
        config = config or RunConfig()
        start_time = datetime.now()

        # Get examples
        if config.subset_size:
            examples = benchmark.sample(config.subset_size, seed=config.random_seed)
        else:
            examples = list(benchmark)

        # Run evaluation
        calculator = MetricsCalculator(dataset_name=benchmark.name)
        errors: list[dict[str, Any]] = []

        iterator = tqdm(examples, desc=f"Evaluating {benchmark.name}") if not config.verbose else examples

        for example in iterator:
            try:
                prediction = self._evaluate_single(example, config)
                calculator.add_prediction(
                    prediction=prediction["answer"],
                    ground_truth=example.answers,
                    question=example.question,
                    reasoning_paths=prediction.get("paths", []),
                    confidence=prediction.get("confidence", 1.0),
                    latency_ms=prediction.get("latency_ms", 0.0),
                    question_id=example.question_id,
                )
            except Exception as e:
                errors.append(
                    {
                        "question_id": example.question_id,
                        "question": example.question,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                )
                # Add empty prediction for failed examples
                calculator.add_prediction(
                    prediction="",
                    ground_truth=example.answers,
                    question=example.question,
                    confidence=0.0,
                    question_id=example.question_id,
                )

        end_time = datetime.now()

        return RunReport(
            benchmark_name=benchmark.name,
            reasoner_name=self.reasoner.name,
            results=calculator.compute(),
            config=config,
            start_time=start_time,
            end_time=end_time,
            errors=errors,
        )

    def _evaluate_single(
        self,
        example: BenchmarkExample,
        config: RunConfig,
    ) -> dict[str, Any]:
        """Evaluate a single example."""
        question = self.pre_process(example)

        # Run reasoner with retries
        last_error = None
        for attempt in range(config.max_retries):
            try:
                result = self.reasoner.reason(question)

                answer = self.post_process(result.primary_answer)

                return {
                    "answer": answer,
                    "confidence": result.confidence,
                    "latency_ms": result.latency_ms,
                    "paths": [p.to_text() for p in result.paths],
                }
            except Exception as e:
                last_error = e
                if config.verbose:
                    print(f"  Attempt {attempt + 1} failed: {e}")

        raise last_error or RuntimeError("Unknown error")

    def compare(
        self,
        reasoners: list[Reasoner],
        benchmark: BaseBenchmark,
        config: RunConfig | None = None,
    ) -> list[RunReport]:
        """
        Compare multiple reasoners on the same benchmark.

        Args:
            reasoners: List of reasoners to compare
            benchmark: Benchmark to evaluate on
            config: Evaluation configuration

        Returns:
            List of reports, one per reasoner
        """
        reports = []

        for reasoner in reasoners:
            print(f"\nEvaluating {reasoner.name}...")
            self.reasoner = reasoner
            report = self.evaluate(benchmark, config)
            reports.append(report)
            print(f"  Accuracy: {report.results.accuracy:.2%}")

        return reports


def compare_reports(reports: list[RunReport]) -> str:
    """Generate a comparison table for multiple reports."""
    lines = [
        "=" * 80,
        "Comparison Results",
        "=" * 80,
        f"{'Reasoner':<30} {'Accuracy':>10} {'F1':>10} {'Latency(ms)':>12} {'Errors':>8}",
        "-" * 80,
    ]

    for report in reports:
        lines.append(
            f"{report.reasoner_name:<30} "
            f"{report.results.accuracy:>10.2%} "
            f"{report.results.f1:>10.2%} "
            f"{report.results.metrics.get('avg_latency_ms', 0):>12.0f} "
            f"{len(report.errors):>8}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)
