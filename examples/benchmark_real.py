#!/usr/bin/env python3
"""
Real benchmark using downloadable QA datasets.

This script uses actual benchmark datasets:
- SimpleQuestions (Wikidata version): 100K+ single-hop questions
- LC-QuAD 2.0: 30K Wikidata questions (simple and complex)
- QALD-9: 500+ multilingual questions

Usage:
    # Download and run SimpleQuestions (small subset)
    python examples/benchmark_real.py --dataset simple-questions --num-examples 100
    
    # Run LC-QuAD 2.0
    python examples/benchmark_real.py --dataset lcquad2 --num-examples 100
    
    # Compare reasoners on real data
    python examples/benchmark_real.py --dataset simple-questions --compare-all --num-examples 50
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import httpx

from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.knowledge_graphs.wikidata import WikidataKG
from neurosym_kg.llm_backends.base import BaseLLMBackend
from neurosym_kg.core.types import LLMResponse, Message
from neurosym_kg.reasoners import (
    ThinkOnGraphReasoner,
    ReasoningOnGraphs,
    GraphRAGReasoner,
    SubgraphRAGReasoner,
)


# =============================================================================
# Data Directory
# =============================================================================

DATA_DIR = Path("data/benchmarks")
DATA_DIR.mkdir(parents=True, exist_ok=True)


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
    
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
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
# Dataset Loaders
# =============================================================================

@dataclass
class QAExample:
    """A QA example from a benchmark."""
    question: str
    answers: list[str]
    subject_entity: Optional[str] = None  # Wikidata ID if available
    predicate: Optional[str] = None
    source: str = ""


def download_file(url: str, path: Path) -> Path:
    """Download a file if it doesn't exist."""
    if not path.exists():
        print(f"   Downloading {url}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(url, path)
        print(f"   Saved to {path}")
    return path


def load_simple_questions_wikidata(max_examples: Optional[int] = None) -> list[QAExample]:
    """
    Load SimpleQuestions dataset (Wikidata version).
    
    Source: https://github.com/askplatypus/wikidata-simplequestions
    Format: subject \t predicate \t object \t question
    """
    # URL for SimpleQuestions Wikidata
    url = "https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_test.txt"
    path = DATA_DIR / "simple_questions_wikidata_test.txt"
    
    try:
        download_file(url, path)
    except Exception as e:
        print(f"   Failed to download SimpleQuestions: {e}")
        print("   Using fallback mini dataset...")
        return load_fallback_dataset(max_examples)
    
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                subject, predicate, obj, question = parts[0], parts[1], parts[2], parts[3]
                examples.append(QAExample(
                    question=question,
                    answers=[obj],
                    subject_entity=subject,
                    predicate=predicate,
                    source="SimpleQuestions-Wikidata"
                ))
    
    if max_examples:
        random.shuffle(examples)
        examples = examples[:max_examples]
    
    return examples


def load_lcquad2(max_examples: Optional[int] = None) -> list[QAExample]:
    """
    Load LC-QuAD 2.0 dataset.
    
    Source: http://lc-quad.sda.tech/
    30K questions over Wikidata
    """
    url = "https://raw.githubusercontent.com/AskNowQA/LC-QuAD2.0/master/dataset/test.json"
    path = DATA_DIR / "lcquad2_test.json"
    
    try:
        download_file(url, path)
    except Exception as e:
        print(f"   Failed to download LC-QuAD 2.0: {e}")
        print("   Using fallback mini dataset...")
        return load_fallback_dataset(max_examples)
    
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        question = item.get("question", item.get("NNQT_question", ""))
        # LC-QuAD has SPARQL, we extract answer from it or use paraphrase
        sparql = item.get("sparql_wikidata", "")
        
        # Simple extraction of entity from SPARQL (rough)
        answers = []
        if "label" in item:
            answers = [item["label"]]
        
        if question and not answers:
            # Skip if no answer extractable
            continue
            
        examples.append(QAExample(
            question=question,
            answers=answers,
            source="LC-QuAD-2.0"
        ))
    
    if max_examples:
        random.shuffle(examples)
        examples = examples[:max_examples]
    
    return examples


def load_mintaka(max_examples: Optional[int] = None) -> list[QAExample]:
    """
    Load Mintaka dataset - multilingual, complex QA over Wikidata.
    
    Source: https://github.com/amazon-science/mintaka
    ~20K questions with Wikidata answers
    """
    url = "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_test.json"
    path = DATA_DIR / "mintaka_test.json"
    
    try:
        download_file(url, path)
    except Exception as e:
        print(f"   Failed to download Mintaka: {e}")
        return load_fallback_dataset(max_examples)
    
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        question = item.get("question", "")
        answer_obj = item.get("answer", {})
        
        # Extract answer
        answers = []
        if isinstance(answer_obj, dict):
            if "mention" in answer_obj:
                answers = [answer_obj["mention"]]
            elif "answer" in answer_obj:
                ans = answer_obj["answer"]
                if isinstance(ans, list):
                    answers = [str(a.get("mention", a)) if isinstance(a, dict) else str(a) for a in ans]
                else:
                    answers = [str(ans)]
        
        if question and answers:
            # Get subject entity if available
            subject = None
            if "questionEntity" in item and item["questionEntity"]:
                entities = item["questionEntity"]
                if isinstance(entities, list) and len(entities) > 0:
                    subject = entities[0].get("name", None)
            
            examples.append(QAExample(
                question=question,
                answers=answers,
                subject_entity=subject,
                source="Mintaka"
            ))
    
    if max_examples:
        random.shuffle(examples)
        examples = examples[:max_examples]
    
    return examples


def load_fallback_dataset(max_examples: Optional[int] = None) -> list[QAExample]:
    """Fallback dataset with more questions when downloads fail."""
    
    # Extended list of verifiable questions
    questions = [
        # Countries & Capitals
        ("What is the capital of France?", ["Paris"]),
        ("What is the capital of Germany?", ["Berlin"]),
        ("What is the capital of Japan?", ["Tokyo"]),
        ("What is the capital of Italy?", ["Rome"]),
        ("What is the capital of Spain?", ["Madrid"]),
        ("What is the capital of Canada?", ["Ottawa"]),
        ("What is the capital of Australia?", ["Canberra"]),
        ("What is the capital of Brazil?", ["Brasília", "Brasilia"]),
        ("What is the capital of India?", ["New Delhi"]),
        ("What is the capital of China?", ["Beijing"]),
        ("What is the capital of Russia?", ["Moscow"]),
        ("What is the capital of Mexico?", ["Mexico City"]),
        ("What is the capital of Egypt?", ["Cairo"]),
        ("What is the capital of South Africa?", ["Pretoria", "Cape Town", "Bloemfontein"]),
        ("What is the capital of Argentina?", ["Buenos Aires"]),
        
        # Languages
        ("What language is spoken in Brazil?", ["Portuguese"]),
        ("What language is spoken in Germany?", ["German"]),
        ("What language is spoken in Japan?", ["Japanese"]),
        ("What language is spoken in France?", ["French"]),
        ("What language is spoken in China?", ["Chinese", "Mandarin"]),
        ("What language is spoken in Russia?", ["Russian"]),
        ("What language is spoken in Italy?", ["Italian"]),
        ("What language is spoken in Spain?", ["Spanish"]),
        
        # Currencies
        ("What is the currency of Japan?", ["Yen", "Japanese yen"]),
        ("What is the currency of United Kingdom?", ["Pound", "Pound sterling"]),
        ("What is the currency of India?", ["Rupee", "Indian rupee"]),
        ("What is the currency of China?", ["Yuan", "Renminbi"]),
        ("What is the currency of Russia?", ["Ruble", "Russian ruble"]),
        ("What is the currency of Brazil?", ["Real", "Brazilian real"]),
        ("What is the currency of Mexico?", ["Peso", "Mexican peso"]),
        ("What is the currency of Switzerland?", ["Franc", "Swiss franc"]),
        
        # Movies & Directors
        ("Who directed Inception?", ["Christopher Nolan"]),
        ("Who directed Titanic?", ["James Cameron"]),
        ("Who directed Pulp Fiction?", ["Quentin Tarantino"]),
        ("Who directed The Godfather?", ["Francis Ford Coppola"]),
        ("Who directed Jurassic Park?", ["Steven Spielberg"]),
        ("Who directed Avatar?", ["James Cameron"]),
        ("Who directed The Dark Knight?", ["Christopher Nolan"]),
        ("Who directed Forrest Gump?", ["Robert Zemeckis"]),
        ("Who directed Schindler's List?", ["Steven Spielberg"]),
        ("Who directed Fight Club?", ["David Fincher"]),
        
        # Famous People Birthplaces
        ("Where was Albert Einstein born?", ["Ulm", "Germany"]),
        ("Where was Mozart born?", ["Salzburg", "Austria"]),
        ("Where was Shakespeare born?", ["Stratford-upon-Avon"]),
        ("Where was Leonardo da Vinci born?", ["Vinci", "Italy"]),
        ("Where was Napoleon born?", ["Corsica", "Ajaccio"]),
        ("Where was Beethoven born?", ["Bonn", "Germany"]),
        ("Where was Picasso born?", ["Málaga", "Spain"]),
        ("Where was Gandhi born?", ["Porbandar", "India"]),
        
        # Companies & Founders
        ("Who founded Apple?", ["Steve Jobs", "Steve Wozniak"]),
        ("Who founded Microsoft?", ["Bill Gates", "Paul Allen"]),
        ("Who founded Amazon?", ["Jeff Bezos"]),
        ("Who founded Tesla?", ["Elon Musk", "Martin Eberhard"]),
        ("Who founded Facebook?", ["Mark Zuckerberg"]),
        ("Who founded Google?", ["Larry Page", "Sergey Brin"]),
        
        # Science
        ("Who discovered penicillin?", ["Alexander Fleming"]),
        ("Who invented the telephone?", ["Alexander Graham Bell"]),
        ("Who invented the light bulb?", ["Thomas Edison"]),
        ("Who developed the theory of relativity?", ["Albert Einstein"]),
        ("Who discovered gravity?", ["Isaac Newton"]),
        
        # Geography
        ("What is the largest country by area?", ["Russia"]),
        ("What is the smallest country?", ["Vatican City"]),
        ("What is the longest river?", ["Nile", "Amazon"]),
        ("What is the highest mountain?", ["Mount Everest", "Everest"]),
        ("What is the largest ocean?", ["Pacific Ocean", "Pacific"]),
        ("What continent is Egypt in?", ["Africa"]),
        ("What continent is Brazil in?", ["South America"]),
        ("What continent is Japan in?", ["Asia"]),
        
        # Books & Authors
        ("Who wrote Harry Potter?", ["J.K. Rowling", "J. K. Rowling"]),
        ("Who wrote Romeo and Juliet?", ["William Shakespeare", "Shakespeare"]),
        ("Who wrote 1984?", ["George Orwell"]),
        ("Who wrote The Great Gatsby?", ["F. Scott Fitzgerald"]),
        ("Who wrote Pride and Prejudice?", ["Jane Austen"]),
        
        # Sports
        ("What sport does Cristiano Ronaldo play?", ["Football", "Soccer"]),
        ("What sport does LeBron James play?", ["Basketball"]),
        ("What sport does Roger Federer play?", ["Tennis"]),
        ("What sport does Tiger Woods play?", ["Golf"]),
        ("What country won the 2022 FIFA World Cup?", ["Argentina"]),
        
        # Music
        ("What band was John Lennon in?", ["The Beatles", "Beatles"]),
        ("What band was Freddie Mercury in?", ["Queen"]),
        ("What instrument does Yo-Yo Ma play?", ["Cello"]),
        
        # Miscellaneous
        ("What animal is the symbol of the WWF?", ["Panda", "Giant panda"]),
        ("What is the chemical symbol for gold?", ["Au"]),
        ("What is the chemical symbol for water?", ["H2O"]),
        ("How many planets are in our solar system?", ["8", "Eight"]),
        ("What is the largest planet?", ["Jupiter"]),
    ]
    
    examples = [
        QAExample(question=q, answers=a, source="Fallback")
        for q, a in questions
    ]
    
    if max_examples:
        random.shuffle(examples)
        examples = examples[:max_examples]
    
    return examples


def load_dataset(name: str, max_examples: Optional[int] = None) -> list[QAExample]:
    """Load a dataset by name."""
    loaders = {
        "simple-questions": load_simple_questions_wikidata,
        "lcquad2": load_lcquad2,
        "mintaka": load_mintaka,
        "fallback": load_fallback_dataset,
    }
    
    loader = loaders.get(name.lower())
    if not loader:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(loaders.keys())}")
        return []
    
    return loader(max_examples)


# =============================================================================
# Evaluation
# =============================================================================

def normalize_answer(text: str) -> str:
    """Normalize for comparison."""
    return text.lower().strip().replace("_", " ").replace("-", " ")


def check_answer(predicted: str, expected_list: list[str]) -> tuple[bool, float]:
    """Check if answer matches. Returns (exact_match, f1)."""
    if not predicted:
        return False, 0.0
    
    pred_norm = normalize_answer(predicted)
    
    for exp in expected_list:
        exp_norm = normalize_answer(exp)
        if exp_norm in pred_norm or pred_norm in exp_norm:
            return True, 1.0
    
    # F1 calculation
    pred_tokens = set(pred_norm.split())
    best_f1 = 0.0
    for exp in expected_list:
        exp_tokens = set(normalize_answer(exp).split())
        if pred_tokens and exp_tokens:
            common = pred_tokens & exp_tokens
            p = len(common) / len(pred_tokens)
            r = len(common) / len(exp_tokens)
            if p + r > 0:
                best_f1 = max(best_f1, 2 * p * r / (p + r))
    
    return False, best_f1


def run_evaluation(
    reasoner,
    reasoner_name: str,
    examples: list[QAExample],
    verbose: bool = False,
) -> dict:
    """Run evaluation."""
    
    correct = 0
    total_f1 = 0.0
    total_latency = 0.0
    results = []
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {reasoner_name}")
    print(f"Examples: {len(examples)}")
    print(f"{'='*60}")
    
    for i, ex in enumerate(examples):
        if verbose:
            print(f"\n[{i+1}/{len(examples)}] {ex.question}")
        else:
            print(f"  [{i+1}/{len(examples)}] {ex.question[:45]}...", end=" ", flush=True)
        
        start = time.time()
        try:
            result = reasoner.reason(ex.question)
            latency = (time.time() - start) * 1000
            answer = result.answer if result.answer else ""
            
            is_correct, f1 = check_answer(answer, ex.answers)
            correct += int(is_correct)
            total_f1 += f1
            total_latency += latency
            
            results.append({
                "question": ex.question,
                "predicted": answer[:100],
                "expected": ex.answers,
                "correct": is_correct,
                "f1": f1,
                "latency_ms": latency,
            })
            
            if verbose:
                status = "✅" if is_correct else "❌"
                print(f"  {status} Got: {answer[:50]}")
                print(f"     Expected: {ex.answers[:2]}")
            else:
                print("✅" if is_correct else "❌")
                
        except Exception as e:
            latency = (time.time() - start) * 1000
            total_latency += latency
            results.append({
                "question": ex.question,
                "predicted": f"ERROR: {str(e)[:30]}",
                "expected": ex.answers,
                "correct": False,
                "f1": 0,
                "latency_ms": latency,
            })
            if not verbose:
                print("❌")
    
    n = len(examples)
    return {
        "reasoner": reasoner_name,
        "num_examples": n,
        "correct": correct,
        "accuracy": correct / n if n > 0 else 0,
        "avg_f1": total_f1 / n if n > 0 else 0,
        "avg_latency_ms": total_latency / n if n > 0 else 0,
        "results": results,
    }


def print_summary(report: dict):
    """Print summary."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {report['reasoner']}")
    print(f"{'='*60}")
    print(f"  Examples:     {report['num_examples']}")
    print(f"  Correct:      {report['correct']}")
    print(f"  Accuracy:     {report['accuracy']:.1%}")
    print(f"  Avg F1:       {report['avg_f1']:.3f}")
    print(f"  Avg Latency:  {report['avg_latency_ms']:.0f}ms")
    
    # Show some errors
    errors = [r for r in report['results'] if not r['correct']][:3]
    if errors:
        print(f"\n  Sample Errors:")
        for e in errors:
            print(f"    Q: {e['question'][:50]}...")
            print(f"    Got: {e['predicted'][:40]}...")
            print(f"    Expected: {e['expected'][:2]}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run real QA benchmarks")
    parser.add_argument("--dataset", default="mintaka",
                       choices=["simple-questions", "lcquad2", "mintaka", "fallback"],
                       help="Dataset to use")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples")
    parser.add_argument("--reasoner", default="tog", 
                       choices=["tog", "rog", "graphrag", "subgraphrag"])
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=str, help="Save results to JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    print(f"{'='*60}")
    print("NeuroSym-KG Real Benchmark")
    print(f"{'='*60}")
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model}")
    print(f"Examples:   {args.num_examples}")
    
    # Test Ollama
    print(f"\n1. Testing Ollama...")
    try:
        test_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        resp = test_llm.generate([Message(role="user", content="Say OK")])
        print(f"   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Load dataset
    print(f"\n2. Loading dataset: {args.dataset}")
    examples = load_dataset(args.dataset, args.num_examples)
    print(f"   ✅ Loaded {len(examples)} examples")
    
    if not examples:
        print("   No examples loaded!")
        return
    
    # Initialize KG
    print(f"\n3. Connecting to Wikidata...")
    try:
        kg = WikidataKG()
        print(f"   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Run evaluation
    all_reports = []
    
    if args.compare_all:
        reasoner_configs = [("tog", "ToG"), ("graphrag", "GraphRAG"), ("subgraphrag", "SubgraphRAG")]
    else:
        reasoner_configs = [(args.reasoner, args.reasoner.upper())]
    
    for rkey, rname in reasoner_configs:
        llm = OllamaBackend(model=args.model, base_url=args.base_url)
        kg = WikidataKG()
        
        if rkey == "tog":
            reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm, max_depth=2, beam_width=3, verbose=False)
        elif rkey == "graphrag":
            reasoner = GraphRAGReasoner(kg=kg, llm=llm, verbose=False)
        elif rkey == "subgraphrag":
            reasoner = SubgraphRAGReasoner(kg=kg, llm=llm, subgraph_size=30, verbose=False)
        else:
            continue
        
        report = run_evaluation(reasoner, rname, examples, args.verbose)
        print_summary(report)
        all_reports.append(report)
    
    # Comparison
    if len(all_reports) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"{'Reasoner':<20} {'Accuracy':>10} {'F1':>10} {'Latency':>12}")
        print(f"{'-'*60}")
        for r in all_reports:
            print(f"{r['reasoner']:<20} {r['accuracy']:>10.1%} {r['avg_f1']:>10.3f} {r['avg_latency_ms']:>10.0f}ms")
    
    # Save
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "reports": all_reports}, f, indent=2)
        print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
