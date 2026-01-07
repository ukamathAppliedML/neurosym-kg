#!/usr/bin/env python3
"""
Real benchmark using Wikidata as the knowledge graph.

This script runs evaluation on real questions that can be answered
by querying Wikidata's live SPARQL endpoint.

Prerequisites:
    1. Ollama running: ollama serve
    2. Model pulled: ollama pull qwen2.5-coder:7b
    3. Internet connection (for Wikidata queries)

Usage:
    # Run with defaults
    python examples/benchmark_wikidata.py
    
    # Specify model and number of questions
    python examples/benchmark_wikidata.py --model llama3.2 --num-examples 20
    
    # Compare all reasoners
    python examples/benchmark_wikidata.py --compare-all
    
    # Verbose mode
    python examples/benchmark_wikidata.py --verbose
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

from neurosym_kg import Triple
from neurosym_kg.knowledge_graphs.wikidata import WikidataKG
from neurosym_kg.llm_backends.base import BaseLLMBackend
from neurosym_kg.core.types import LLMResponse, Message
from neurosym_kg.reasoners import (
    ThinkOnGraphReasoner,
    ReasoningOnGraphs,
    GraphRAGReasoner,
    SubgraphRAGReasoner,
)
from neurosym_kg.evaluation import exact_match, f1_score


# =============================================================================
# Ollama Backend
# =============================================================================

class OllamaBackend(BaseLLMBackend):
    """Ollama LLM backend."""
    
    def __init__(
        self, 
        model: str = "qwen2.5-coder:7b", 
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        super().__init__(model=model)
        self._model = model
        self.base_url = base_url
        self.timeout = timeout
        self.call_count = 0
    
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        self.call_count += 1
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self._model, 
                "messages": ollama_messages, 
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 256}
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return LLMResponse(content=data["message"]["content"], model=self._model)
    
    async def agenerate(self, messages: list[Message], **kwargs) -> LLMResponse:
        return self.generate(messages, **kwargs)


# =============================================================================
# Real Benchmark Questions (answerable via Wikidata)
# =============================================================================

@dataclass
class BenchmarkQuestion:
    """A benchmark question with ground truth."""
    question: str
    answers: list[str]
    wikidata_entity: str  # Starting entity for KG traversal
    question_type: str  # 1-hop, 2-hop, multi-answer
    difficulty: str  # easy, medium, hard


# These are real questions that can be answered by Wikidata
WIKIDATA_BENCHMARK = [
    # === EASY: 1-hop questions ===
    BenchmarkQuestion(
        question="What is the capital of France?",
        answers=["Paris"],
        wikidata_entity="Q142",  # France
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What is the capital of Japan?",
        answers=["Tokyo"],
        wikidata_entity="Q17",  # Japan
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What is the capital of Germany?",
        answers=["Berlin"],
        wikidata_entity="Q183",  # Germany
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What is the capital of Italy?",
        answers=["Rome"],
        wikidata_entity="Q38",  # Italy
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What is the capital of Spain?",
        answers=["Madrid"],
        wikidata_entity="Q29",  # Spain
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="Who directed Inception?",
        answers=["Christopher Nolan"],
        wikidata_entity="Q25188",  # Inception
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="Who wrote Harry Potter?",
        answers=["J. K. Rowling", "J.K. Rowling", "Rowling"],
        wikidata_entity="Q8337",  # Harry Potter
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What country is Berlin in?",
        answers=["Germany"],
        wikidata_entity="Q64",  # Berlin
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What language is spoken in Brazil?",
        answers=["Portuguese", "Brazilian Portuguese"],
        wikidata_entity="Q155",  # Brazil
        question_type="1-hop",
        difficulty="easy",
    ),
    BenchmarkQuestion(
        question="What is the currency of Japan?",
        answers=["Japanese yen", "yen", "JPY"],
        wikidata_entity="Q17",  # Japan
        question_type="1-hop",
        difficulty="easy",
    ),
    
    # === MEDIUM: 2-hop questions ===
    BenchmarkQuestion(
        question="Where was the director of Inception born?",
        answers=["London", "Westminster"],
        wikidata_entity="Q25188",  # Inception
        question_type="2-hop",
        difficulty="medium",
    ),
    BenchmarkQuestion(
        question="What country was Albert Einstein born in?",
        answers=["Germany", "German Empire", "Ulm"],
        wikidata_entity="Q937",  # Einstein
        question_type="2-hop",
        difficulty="medium",
    ),
    BenchmarkQuestion(
        question="What is the capital of the country where the Eiffel Tower is located?",
        answers=["Paris"],
        wikidata_entity="Q243",  # Eiffel Tower
        question_type="2-hop",
        difficulty="medium",
    ),
    BenchmarkQuestion(
        question="Who is the spouse of Barack Obama?",
        answers=["Michelle Obama"],
        wikidata_entity="Q76",  # Barack Obama
        question_type="1-hop",
        difficulty="medium",
    ),
    BenchmarkQuestion(
        question="What nationality is the author of Harry Potter?",
        answers=["British", "United Kingdom", "English"],
        wikidata_entity="Q8337",  # Harry Potter
        question_type="2-hop",
        difficulty="medium",
    ),
    
    # === HARD: Complex questions ===
    BenchmarkQuestion(
        question="What movies did Leonardo DiCaprio star in that were directed by Martin Scorsese?",
        answers=["The Wolf of Wall Street", "Shutter Island", "The Departed", "Gangs of New York", "The Aviator", "Killers of the Flower Moon"],
        wikidata_entity="Q38111",  # Leonardo DiCaprio
        question_type="multi-answer",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        question="What is the population of the capital of France?",
        answers=["2161000", "2.16 million", "2 million"],
        wikidata_entity="Q142",  # France
        question_type="2-hop",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        question="Who founded the company that makes the iPhone?",
        answers=["Steve Jobs", "Steve Wozniak", "Ronald Wayne"],
        wikidata_entity="Q2766",  # iPhone
        question_type="2-hop",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        question="What university did the founder of Microsoft attend?",
        answers=["Harvard", "Harvard University"],
        wikidata_entity="Q2283",  # Microsoft
        question_type="2-hop",
        difficulty="hard",
    ),
    BenchmarkQuestion(
        question="In what country is the headquarters of Toyota located?",
        answers=["Japan"],
        wikidata_entity="Q53268",  # Toyota
        question_type="2-hop",
        difficulty="hard",
    ),
]


# =============================================================================
# Evaluation
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    return text.lower().strip().replace("_", " ").replace("-", " ")


def check_answer(predicted: str, expected_list: list[str]) -> tuple[bool, float]:
    """Check if predicted answer matches expected. Returns (exact_match, f1)."""
    pred_norm = normalize_answer(predicted)
    
    # Check exact match
    for exp in expected_list:
        exp_norm = normalize_answer(exp)
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True, 1.0
    
    # Calculate F1
    pred_tokens = set(pred_norm.split())
    best_f1 = 0.0
    for exp in expected_list:
        exp_tokens = set(normalize_answer(exp).split())
        if not pred_tokens or not exp_tokens:
            continue
        common = pred_tokens & exp_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(exp_tokens) if exp_tokens else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    
    return False, best_f1


def run_evaluation(
    reasoner,
    reasoner_name: str,
    questions: list[BenchmarkQuestion],
    verbose: bool = False,
) -> dict:
    """Run evaluation on questions."""
    
    results = []
    correct = 0
    total_f1 = 0.0
    total_latency = 0.0
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {reasoner_name}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*60}")
    
    for i, q in enumerate(questions):
        if verbose:
            print(f"\n[{i+1}/{len(questions)}] {q.question}")
        else:
            print(f"  [{i+1}/{len(questions)}] {q.question[:50]}...", end=" ", flush=True)
        
        start_time = time.time()
        try:
            result = reasoner.reason(q.question)
            latency_ms = (time.time() - start_time) * 1000
            answer = result.answer if result.answer else "(no answer)"
            
            is_correct, f1 = check_answer(answer, q.answers)
            
            if is_correct:
                correct += 1
            total_f1 += f1
            total_latency += latency_ms
            
            results.append({
                "question": q.question,
                "predicted": answer,
                "expected": q.answers,
                "correct": is_correct,
                "f1": f1,
                "latency_ms": latency_ms,
                "difficulty": q.difficulty,
                "type": q.question_type,
            })
            
            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"  {status} Predicted: {answer[:50]}")
                print(f"     Expected: {q.answers[:2]}")
                print(f"     F1: {f1:.2f}, Latency: {latency_ms:.0f}ms")
            else:
                print("✅" if is_correct else "❌")
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            total_latency += latency_ms
            
            results.append({
                "question": q.question,
                "predicted": f"ERROR: {str(e)[:50]}",
                "expected": q.answers,
                "correct": False,
                "f1": 0,
                "latency_ms": latency_ms,
                "difficulty": q.difficulty,
                "type": q.question_type,
            })
            
            if verbose:
                print(f"  ❌ Error: {str(e)[:50]}")
            else:
                print("❌")
    
    num_questions = len(questions)
    accuracy = correct / num_questions if num_questions > 0 else 0
    avg_f1 = total_f1 / num_questions if num_questions > 0 else 0
    avg_latency = total_latency / num_questions if num_questions > 0 else 0
    
    return {
        "reasoner": reasoner_name,
        "num_questions": num_questions,
        "correct": correct,
        "accuracy": accuracy,
        "avg_f1": avg_f1,
        "avg_latency_ms": avg_latency,
        "results": results,
    }


def print_summary(report: dict):
    """Print evaluation summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {report['reasoner']}")
    print(f"{'='*60}")
    print(f"  Questions:    {report['num_questions']}")
    print(f"  Correct:      {report['correct']}")
    print(f"  Accuracy:     {report['accuracy']:.1%}")
    print(f"  Avg F1:       {report['avg_f1']:.3f}")
    print(f"  Avg Latency:  {report['avg_latency_ms']:.0f}ms")
    
    # Breakdown by difficulty
    by_difficulty = {}
    for r in report['results']:
        d = r['difficulty']
        if d not in by_difficulty:
            by_difficulty[d] = {'correct': 0, 'total': 0}
        by_difficulty[d]['total'] += 1
        if r['correct']:
            by_difficulty[d]['correct'] += 1
    
    print(f"\n  By Difficulty:")
    for diff in ['easy', 'medium', 'hard']:
        if diff in by_difficulty:
            d = by_difficulty[diff]
            acc = d['correct'] / d['total'] if d['total'] > 0 else 0
            print(f"    {diff:8}: {d['correct']}/{d['total']} ({acc:.0%})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run real benchmark with Wikidata")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama URL")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of questions")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "all"], default="all")
    parser.add_argument("--reasoner", default="tog", choices=["tog", "rog", "graphrag", "subgraphrag"])
    parser.add_argument("--compare-all", action="store_true", help="Compare all reasoners")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("NeuroSym-KG Real Benchmark (Wikidata)")
    print(f"{'='*60}")
    print(f"Model:      {args.model}")
    print(f"Questions:  {args.num_examples}")
    print(f"Difficulty: {args.difficulty}")
    
    # Test Ollama
    print(f"\n1. Testing Ollama...")
    try:
        test_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        resp = test_llm.generate([Message(role="user", content="Say OK")])
        print(f"   ✅ Connected: {resp.content.strip()[:20]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test Wikidata
    print(f"\n2. Testing Wikidata connection...")
    try:
        kg = WikidataKG()
        # Quick test query
        entity = kg.get_entity("Q142")  # France
        print(f"   ✅ Connected: Retrieved Q142 ({entity.name if entity else 'France'})")
    except Exception as e:
        print(f"   ❌ Wikidata connection failed: {e}")
        print("   Check your internet connection")
        return
    
    # Filter questions
    questions = WIKIDATA_BENCHMARK.copy()
    if args.difficulty != "all":
        questions = [q for q in questions if q.difficulty == args.difficulty]
    questions = questions[:args.num_examples]
    
    print(f"\n3. Selected {len(questions)} questions")
    
    # Run evaluation
    all_reports = []
    
    if args.compare_all:
        reasoner_configs = [
            ("tog", "Think-on-Graph"),
            ("rog", "Reasoning on Graphs"),
            ("graphrag", "GraphRAG"),
            ("subgraphrag", "SubgraphRAG"),
        ]
    else:
        reasoner_map = {
            "tog": "Think-on-Graph",
            "rog": "Reasoning on Graphs", 
            "graphrag": "GraphRAG",
            "subgraphrag": "SubgraphRAG",
        }
        reasoner_configs = [(args.reasoner, reasoner_map[args.reasoner])]
    
    for rkey, rname in reasoner_configs:
        # Fresh LLM and KG for each reasoner
        llm = OllamaBackend(model=args.model, base_url=args.base_url)
        kg = WikidataKG()
        
        if rkey == "tog":
            reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm, max_depth=2, beam_width=3, verbose=False)
        elif rkey == "rog":
            reasoner = ReasoningOnGraphs(kg=kg, llm=llm, max_path_length=3, num_plans=2, verbose=False)
        elif rkey == "graphrag":
            reasoner = GraphRAGReasoner(kg=kg, llm=llm, verbose=False)
        elif rkey == "subgraphrag":
            reasoner = SubgraphRAGReasoner(kg=kg, llm=llm, subgraph_size=30, verbose=False)
        
        report = run_evaluation(reasoner, rname, questions, verbose=args.verbose)
        print_summary(report)
        all_reports.append(report)
    
    # Comparison table
    if len(all_reports) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"{'Reasoner':<25} {'Accuracy':>10} {'F1':>10} {'Latency':>12}")
        print(f"{'-'*70}")
        for r in all_reports:
            print(f"{r['reasoner']:<25} {r['accuracy']:>10.1%} {r['avg_f1']:>10.3f} {r['avg_latency_ms']:>10.0f}ms")
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "reports": all_reports,
            }, f, indent=2)
        print(f"\n✅ Saved to {args.output}")
    
    print(f"\n{'='*60}")
    print("Benchmark complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
