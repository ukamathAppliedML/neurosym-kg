"""
Benchmark dataset loaders for evaluation.

Supports standard KG-QA benchmarks:
- WebQSP (WebQuestionsSP)
- CWQ (ComplexWebQuestions)
- MetaQA
- SimpleQuestions
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import random


@dataclass
class BenchmarkExample:
    """A single example from a benchmark dataset."""

    question: str
    answers: list[str]
    question_id: str = ""
    topic_entity: str = ""
    topic_entity_id: str = ""
    sparql: str = ""  # Gold SPARQL query if available
    reasoning_type: str = ""  # e.g., "1-hop", "2-hop", "comparison"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBenchmark(ABC):
    """Abstract base class for benchmark datasets."""

    def __init__(
        self,
        name: str,
        data_dir: str | Path | None = None,
    ) -> None:
        self.name = name
        self.data_dir = Path(data_dir) if data_dir else None
        self._examples: list[BenchmarkExample] | None = None

    @property
    def examples(self) -> list[BenchmarkExample]:
        """Lazy load examples."""
        if self._examples is None:
            self._examples = list(self._load_examples())
        return self._examples

    @abstractmethod
    def _load_examples(self) -> Iterator[BenchmarkExample]:
        """Load examples from the dataset."""
        ...

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[BenchmarkExample]:
        return iter(self.examples)

    def __getitem__(self, idx: int) -> BenchmarkExample:
        return self.examples[idx]

    def sample(self, n: int, seed: int | None = None) -> list[BenchmarkExample]:
        """Sample n examples from the dataset."""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.examples, min(n, len(self.examples)))

    def filter_by_type(self, reasoning_type: str) -> list[BenchmarkExample]:
        """Filter examples by reasoning type."""
        return [ex for ex in self.examples if ex.reasoning_type == reasoning_type]


class WebQSP(BaseBenchmark):
    """
    WebQuestionsSP (Semantic Parsing) benchmark.

    A subset of WebQuestions with SPARQL annotations.
    Contains ~4,737 questions with Freebase annotations.

    Download from: https://www.microsoft.com/en-us/download/details.aspx?id=52763
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "test",
    ) -> None:
        super().__init__("WebQSP", data_dir)
        self.split = split

    def _load_examples(self) -> Iterator[BenchmarkExample]:
        """Load WebQSP examples."""
        if self.data_dir is None:
            # Return demo examples if no data directory
            yield from self._demo_examples()
            return

        file_path = self.data_dir / f"WebQSP.{self.split}.json"
        if not file_path.exists():
            yield from self._demo_examples()
            return

        with open(file_path) as f:
            data = json.load(f)

        for item in data.get("Questions", []):
            # Extract answers
            answers = []
            for parse in item.get("Parses", []):
                for ans in parse.get("Answers", []):
                    ans_text = ans.get("AnswerArgument", "") or ans.get("EntityName", "")
                    if ans_text and ans_text not in answers:
                        answers.append(ans_text)

            # Get topic entity
            topic_entity = ""
            topic_entity_id = ""
            if item.get("Parses"):
                topic_entity_id = item["Parses"][0].get("TopicEntityMid", "")
                topic_entity = item["Parses"][0].get("TopicEntityName", "")

            # Determine reasoning type based on SPARQL complexity
            sparql = item.get("Parses", [{}])[0].get("Sparql", "")
            reasoning_type = self._classify_reasoning(sparql)

            yield BenchmarkExample(
                question=item.get("RawQuestion", ""),
                answers=answers,
                question_id=item.get("QuestionId", ""),
                topic_entity=topic_entity,
                topic_entity_id=topic_entity_id,
                sparql=sparql,
                reasoning_type=reasoning_type,
            )

    def _classify_reasoning(self, sparql: str) -> str:
        """Classify reasoning type from SPARQL."""
        if not sparql:
            return "unknown"

        # Count joins (approximate hop count)
        join_count = sparql.lower().count(" . ")
        if join_count <= 1:
            return "1-hop"
        elif join_count <= 2:
            return "2-hop"
        else:
            return "multi-hop"

    def _demo_examples(self) -> Iterator[BenchmarkExample]:
        """Return demo examples for testing without data."""
        demo_data = [
            {
                "question": "What is the capital of France?",
                "answers": ["Paris"],
                "topic_entity": "France",
                "reasoning_type": "1-hop",
            },
            {
                "question": "Who directed Inception?",
                "answers": ["Christopher Nolan"],
                "topic_entity": "Inception",
                "reasoning_type": "1-hop",
            },
            {
                "question": "What country is the Eiffel Tower in?",
                "answers": ["France"],
                "topic_entity": "Eiffel Tower",
                "reasoning_type": "1-hop",
            },
            {
                "question": "Who is the spouse of Barack Obama?",
                "answers": ["Michelle Obama"],
                "topic_entity": "Barack Obama",
                "reasoning_type": "1-hop",
            },
            {
                "question": "What movies did Christopher Nolan direct?",
                "answers": ["Inception", "The Dark Knight", "Interstellar", "Dunkirk", "Tenet", "Oppenheimer"],
                "topic_entity": "Christopher Nolan",
                "reasoning_type": "1-hop",
            },
            {
                "question": "Where was the director of Inception born?",
                "answers": ["London", "Westminster"],
                "topic_entity": "Inception",
                "reasoning_type": "2-hop",
            },
            {
                "question": "What university did the CEO of Tesla attend?",
                "answers": ["University of Pennsylvania", "Queen's University", "Stanford University"],
                "topic_entity": "Tesla",
                "reasoning_type": "2-hop",
            },
            {
                "question": "What language is spoken in the country where the Eiffel Tower is located?",
                "answers": ["French"],
                "topic_entity": "Eiffel Tower",
                "reasoning_type": "2-hop",
            },
        ]

        for i, item in enumerate(demo_data):
            yield BenchmarkExample(
                question=item["question"],
                answers=item["answers"],
                question_id=f"demo_{i}",
                topic_entity=item.get("topic_entity", ""),
                reasoning_type=item.get("reasoning_type", "unknown"),
            )


class CWQ(BaseBenchmark):
    """
    ComplexWebQuestions benchmark.

    Contains ~34,689 complex questions requiring multi-hop reasoning.
    Based on WebQuestionsSP with added complexity.

    Download from: https://www.tau-nlp.org/compwebq
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "test",
    ) -> None:
        super().__init__("ComplexWebQuestions", data_dir)
        self.split = split

    def _load_examples(self) -> Iterator[BenchmarkExample]:
        """Load CWQ examples."""
        if self.data_dir is None:
            yield from self._demo_examples()
            return

        file_path = self.data_dir / f"ComplexWebQuestions_{self.split}.json"
        if not file_path.exists():
            yield from self._demo_examples()
            return

        with open(file_path) as f:
            data = json.load(f)

        for item in data:
            answers = []
            for ans in item.get("answers", []):
                ans_text = ans.get("answer", "") or ans.get("entity_name", "")
                if ans_text:
                    answers.append(ans_text)

            yield BenchmarkExample(
                question=item.get("question", ""),
                answers=answers,
                question_id=item.get("ID", ""),
                sparql=item.get("sparql", ""),
                reasoning_type=item.get("compositionality_type", "unknown"),
                metadata={
                    "machine_question": item.get("machine_question", ""),
                    "webqsp_id": item.get("webqsp_ID", ""),
                },
            )

    def _demo_examples(self) -> Iterator[BenchmarkExample]:
        """Demo examples for testing."""
        demo_data = [
            {
                "question": "What movies were directed by the person who directed The Dark Knight?",
                "answers": ["Inception", "Interstellar", "Dunkirk", "Tenet", "Memento"],
                "reasoning_type": "composition",
            },
            {
                "question": "Which country has both the Eiffel Tower and the Louvre?",
                "answers": ["France"],
                "reasoning_type": "conjunction",
            },
            {
                "question": "What is the population of the city where Apple headquarters is located?",
                "answers": ["60000", "60614"],  # Cupertino population
                "reasoning_type": "composition",
            },
            {
                "question": "Name a movie that stars both Leonardo DiCaprio and Tom Hardy",
                "answers": ["Inception", "The Revenant"],
                "reasoning_type": "conjunction",
            },
            {
                "question": "What nationality is the author of Harry Potter?",
                "answers": ["British", "English"],
                "reasoning_type": "composition",
            },
        ]

        for i, item in enumerate(demo_data):
            yield BenchmarkExample(
                question=item["question"],
                answers=item["answers"],
                question_id=f"cwq_demo_{i}",
                reasoning_type=item.get("reasoning_type", "unknown"),
            )


class MetaQA(BaseBenchmark):
    """
    MetaQA benchmark for movie domain QA.

    Contains questions with 1-hop, 2-hop, and 3-hop reasoning.
    ~400K questions over a movie knowledge base.

    Download from: https://github.com/yuyuz/MetaQA
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        hops: int = 1,
        split: str = "test",
    ) -> None:
        super().__init__(f"MetaQA-{hops}hop", data_dir)
        self.hops = hops
        self.split = split

    def _load_examples(self) -> Iterator[BenchmarkExample]:
        """Load MetaQA examples."""
        if self.data_dir is None:
            yield from self._demo_examples()
            return

        file_path = self.data_dir / f"{self.hops}-hop" / f"qa_{self.split}.txt"
        if not file_path.exists():
            yield from self._demo_examples()
            return

        with open(file_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    question = parts[0]
                    answers = parts[1].split("|")

                    # Extract topic entity (usually in brackets)
                    topic = ""
                    if "[" in question and "]" in question:
                        topic = question[question.find("[") + 1 : question.find("]")]

                    yield BenchmarkExample(
                        question=question,
                        answers=answers,
                        question_id=f"metaqa_{self.hops}_{i}",
                        topic_entity=topic,
                        reasoning_type=f"{self.hops}-hop",
                    )

    def _demo_examples(self) -> Iterator[BenchmarkExample]:
        """Demo examples for testing."""
        if self.hops == 1:
            demo_data = [
                {"question": "What movies did [Tom Hanks] star in?", "answers": ["Forrest Gump", "Cast Away", "Saving Private Ryan"]},
                {"question": "Who directed [The Matrix]?", "answers": ["The Wachowskis"]},
                {"question": "What year was [Titanic] released?", "answers": ["1997"]},
            ]
        elif self.hops == 2:
            demo_data = [
                {"question": "Who are the actors in the movies directed by [Steven Spielberg]?", "answers": ["Tom Hanks", "Harrison Ford"]},
                {"question": "What genre are the movies starring [Brad Pitt]?", "answers": ["Drama", "Action", "Thriller"]},
            ]
        else:
            demo_data = [
                {"question": "What awards did actors in [Inception] win?", "answers": ["Oscar", "Golden Globe"]},
            ]

        for i, item in enumerate(demo_data):
            topic = ""
            q = item["question"]
            if "[" in q and "]" in q:
                topic = q[q.find("[") + 1 : q.find("]")]

            yield BenchmarkExample(
                question=item["question"],
                answers=item["answers"],
                question_id=f"metaqa_demo_{self.hops}_{i}",
                topic_entity=topic,
                reasoning_type=f"{self.hops}-hop",
            )


class SimpleQuestions(BaseBenchmark):
    """
    SimpleQuestions benchmark.

    Contains ~100K simple single-relation questions.
    Good baseline for testing basic KG-QA.

    Download from: https://github.com/davidgolub/SimpleQA
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "test",
    ) -> None:
        super().__init__("SimpleQuestions", data_dir)
        self.split = split

    def _load_examples(self) -> Iterator[BenchmarkExample]:
        """Load SimpleQuestions examples."""
        if self.data_dir is None:
            yield from self._demo_examples()
            return

        file_path = self.data_dir / f"annotated_fb_data_{self.split}.txt"
        if not file_path.exists():
            yield from self._demo_examples()
            return

        with open(file_path) as f:
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    subject = parts[0]
                    relation = parts[1]
                    obj = parts[2]
                    question = parts[3]

                    yield BenchmarkExample(
                        question=question,
                        answers=[obj],
                        question_id=f"sq_{i}",
                        topic_entity_id=subject,
                        reasoning_type="1-hop",
                        metadata={"relation": relation},
                    )

    def _demo_examples(self) -> Iterator[BenchmarkExample]:
        """Demo examples."""
        demo_data = [
            {"question": "What is the capital of Germany?", "answers": ["Berlin"]},
            {"question": "Who wrote Romeo and Juliet?", "answers": ["William Shakespeare"]},
            {"question": "What is the currency of Japan?", "answers": ["Yen", "Japanese yen"]},
            {"question": "What language is spoken in Brazil?", "answers": ["Portuguese"]},
        ]

        for i, item in enumerate(demo_data):
            yield BenchmarkExample(
                question=item["question"],
                answers=item["answers"],
                question_id=f"sq_demo_{i}",
                reasoning_type="1-hop",
            )


# Registry of available benchmarks
BENCHMARKS = {
    "webqsp": WebQSP,
    "cwq": CWQ,
    "metaqa-1hop": lambda **kw: MetaQA(hops=1, **kw),
    "metaqa-2hop": lambda **kw: MetaQA(hops=2, **kw),
    "metaqa-3hop": lambda **kw: MetaQA(hops=3, **kw),
    "simplequestions": SimpleQuestions,
}


def load_benchmark(
    name: str,
    data_dir: str | Path | None = None,
    **kwargs: Any,
) -> BaseBenchmark:
    """
    Load a benchmark by name.

    Args:
        name: Benchmark name (webqsp, cwq, metaqa-1hop, etc.)
        data_dir: Directory containing benchmark data
        **kwargs: Additional arguments for the benchmark

    Returns:
        Loaded benchmark
    """
    name_lower = name.lower()
    if name_lower not in BENCHMARKS:
        available = ", ".join(BENCHMARKS.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")

    benchmark_cls = BENCHMARKS[name_lower]
    return benchmark_cls(data_dir=data_dir, **kwargs)
