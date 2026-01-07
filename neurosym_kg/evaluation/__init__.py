"""
Evaluation module for NeuroSym-KG.

Provides benchmarks, metrics, and evaluation utilities for
measuring reasoner performance.
"""

from neurosym_kg.evaluation.benchmarks import (
    BENCHMARKS,
    BaseBenchmark,
    BenchmarkExample,
    CWQ,
    MetaQA,
    SimpleQuestions,
    WebQSP,
    load_benchmark,
)
from neurosym_kg.evaluation.metrics import (
    EvaluationResult,
    MetricsCalculator,
    SinglePrediction,
    exact_match,
    f1_score,
    hits_at_k,
    mean_reciprocal_rank,
    normalize_answer,
)
from neurosym_kg.evaluation.runner import (
    BenchmarkRunner,
    RunConfig,
    RunReport,
    compare_reports,
)

__all__ = [
    # Benchmarks
    "BaseBenchmark",
    "BenchmarkExample",
    "WebQSP",
    "CWQ",
    "MetaQA",
    "SimpleQuestions",
    "BENCHMARKS",
    "load_benchmark",
    # Metrics
    "exact_match",
    "f1_score",
    "hits_at_k",
    "mean_reciprocal_rank",
    "normalize_answer",
    "MetricsCalculator",
    "SinglePrediction",
    "EvaluationResult",
    # Runner
    "BenchmarkRunner",
    "RunConfig",
    "RunReport",
    "compare_reports",
]
