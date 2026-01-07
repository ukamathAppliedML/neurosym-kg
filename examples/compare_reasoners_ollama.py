#!/usr/bin/env python3
"""
Comprehensive demo comparing all reasoning paradigms with Ollama.

This script compares:
- Think-on-Graph (ToG): Beam search with LLM as agent
- Reasoning on Graphs (RoG): Plan-based faithful reasoning  
- GraphRAG: Community-based retrieval
- SubgraphRAG: Flexible subgraph retrieval

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull llama3.2

Usage:
    python examples/compare_reasoners_ollama.py
    
    # Use a different model:
    python examples/compare_reasoners_ollama.py --model qwen2.5-coder:7b
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.llm_backends.base import BaseLLMBackend
from neurosym_kg.core.types import LLMResponse, Message
from neurosym_kg.reasoners import (
    ThinkOnGraphReasoner,
    ReasoningOnGraphs,
    GraphRAGReasoner,
    SubgraphRAGReasoner,
)


# =============================================================================
# Ollama Backend
# =============================================================================

class OllamaBackend(BaseLLMBackend):
    """Ollama LLM backend for local model inference."""
    
    def __init__(
        self, 
        model: str = "llama3.2", 
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ):
        super().__init__(model=model)
        self._model = model
        self.base_url = base_url
        self.timeout = timeout
        self._call_count = 0
    
    def generate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Generate a response using Ollama."""
        self._call_count += 1
        
        ollama_messages = [
            {"role": m.role, "content": m.content} 
            for m in messages
        ]
        
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self._model, 
                "messages": ollama_messages, 
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistency
                    "num_predict": 256,  # Limit response length
                }
            },
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["message"]["content"], 
            model=self._model
        )
    
    async def agenerate(self, messages: list[Message], **kwargs) -> LLMResponse:
        """Async generation (uses sync for simplicity)."""
        return self.generate(messages, **kwargs)
    
    def reset_count(self):
        """Reset the call counter."""
        self._call_count = 0
    
    @property
    def call_count(self) -> int:
        return self._call_count


# =============================================================================
# Knowledge Graph Creation
# =============================================================================

def create_movie_kg() -> InMemoryKG:
    """Create a movie knowledge graph for testing."""
    kg = InMemoryKG(name="Movies KG")
    
    kg.add_triples([
        # Inception
        Triple("Inception", "type", "Movie"),
        Triple("Inception", "director", "Christopher_Nolan"),
        Triple("Inception", "year", "2010"),
        Triple("Inception", "genre", "Science_Fiction"),
        Triple("Inception", "genre", "Thriller"),
        Triple("Inception", "starring", "Leonardo_DiCaprio"),
        Triple("Inception", "starring", "Tom_Hardy"),
        Triple("Inception", "starring", "Ellen_Page"),
        Triple("Inception", "budget", "160_million_USD"),
        Triple("Inception", "box_office", "836_million_USD"),
        
        # The Dark Knight
        Triple("The_Dark_Knight", "type", "Movie"),
        Triple("The_Dark_Knight", "director", "Christopher_Nolan"),
        Triple("The_Dark_Knight", "year", "2008"),
        Triple("The_Dark_Knight", "genre", "Action"),
        Triple("The_Dark_Knight", "genre", "Superhero"),
        Triple("The_Dark_Knight", "starring", "Christian_Bale"),
        Triple("The_Dark_Knight", "starring", "Heath_Ledger"),
        Triple("The_Dark_Knight", "starring", "Aaron_Eckhart"),
        
        # Interstellar
        Triple("Interstellar", "type", "Movie"),
        Triple("Interstellar", "director", "Christopher_Nolan"),
        Triple("Interstellar", "year", "2014"),
        Triple("Interstellar", "genre", "Science_Fiction"),
        Triple("Interstellar", "starring", "Matthew_McConaughey"),
        Triple("Interstellar", "starring", "Anne_Hathaway"),
        
        # Titanic (different director for comparison)
        Triple("Titanic", "type", "Movie"),
        Triple("Titanic", "director", "James_Cameron"),
        Triple("Titanic", "year", "1997"),
        Triple("Titanic", "genre", "Drama"),
        Triple("Titanic", "genre", "Romance"),
        Triple("Titanic", "starring", "Leonardo_DiCaprio"),
        Triple("Titanic", "starring", "Kate_Winslet"),
        Triple("Titanic", "box_office", "2_billion_USD"),
        
        # Avatar
        Triple("Avatar", "type", "Movie"),
        Triple("Avatar", "director", "James_Cameron"),
        Triple("Avatar", "year", "2009"),
        Triple("Avatar", "genre", "Science_Fiction"),
        Triple("Avatar", "starring", "Sam_Worthington"),
        Triple("Avatar", "starring", "Zoe_Saldana"),
        
        # Director information
        Triple("Christopher_Nolan", "type", "Person"),
        Triple("Christopher_Nolan", "occupation", "Director"),
        Triple("Christopher_Nolan", "born_in", "London"),
        Triple("Christopher_Nolan", "birth_year", "1970"),
        Triple("Christopher_Nolan", "nationality", "British"),
        
        Triple("James_Cameron", "type", "Person"),
        Triple("James_Cameron", "occupation", "Director"),
        Triple("James_Cameron", "born_in", "Kapuskasing"),
        Triple("James_Cameron", "birth_year", "1954"),
        Triple("James_Cameron", "nationality", "Canadian"),
        
        # Actor information
        Triple("Leonardo_DiCaprio", "type", "Person"),
        Triple("Leonardo_DiCaprio", "occupation", "Actor"),
        Triple("Leonardo_DiCaprio", "born_in", "Los_Angeles"),
        Triple("Leonardo_DiCaprio", "birth_year", "1974"),
        Triple("Leonardo_DiCaprio", "award", "Academy_Award_2016"),
        
        Triple("Tom_Hardy", "type", "Person"),
        Triple("Tom_Hardy", "occupation", "Actor"),
        Triple("Tom_Hardy", "born_in", "London"),
        
        Triple("Christian_Bale", "type", "Person"),
        Triple("Christian_Bale", "occupation", "Actor"),
        Triple("Christian_Bale", "born_in", "Haverfordwest"),
        Triple("Christian_Bale", "award", "Academy_Award_2011"),
        
        Triple("Heath_Ledger", "type", "Person"),
        Triple("Heath_Ledger", "occupation", "Actor"),
        Triple("Heath_Ledger", "born_in", "Perth"),
        Triple("Heath_Ledger", "award", "Academy_Award_2009"),
        
        # Location hierarchy
        Triple("London", "located_in", "United_Kingdom"),
        Triple("London", "type", "City"),
        Triple("Los_Angeles", "located_in", "California"),
        Triple("Los_Angeles", "type", "City"),
        Triple("California", "located_in", "United_States"),
        Triple("United_Kingdom", "type", "Country"),
        Triple("United_States", "type", "Country"),
    ])
    
    return kg


# =============================================================================
# Test Questions
# =============================================================================

@dataclass
class TestQuestion:
    """A test question with expected answers."""
    question: str
    expected_answers: list[str]
    question_type: str  # 1-hop, 2-hop, aggregation, comparison


TEST_QUESTIONS = [
    # 1-hop questions (direct lookup)
    TestQuestion(
        question="Who directed Inception?",
        expected_answers=["Christopher Nolan", "Christopher_Nolan"],
        question_type="1-hop",
    ),
    TestQuestion(
        question="What year was The Dark Knight released?",
        expected_answers=["2008"],
        question_type="1-hop",
    ),
    TestQuestion(
        question="What genre is Interstellar?",
        expected_answers=["Science Fiction", "Science_Fiction", "Sci-Fi"],
        question_type="1-hop",
    ),
    
    # 2-hop questions (require traversal)
    TestQuestion(
        question="Where was the director of Inception born?",
        expected_answers=["London"],
        question_type="2-hop",
    ),
    TestQuestion(
        question="What is the nationality of the director of Titanic?",
        expected_answers=["Canadian"],
        question_type="2-hop",
    ),
    
    # Aggregation questions
    TestQuestion(
        question="What movies did Christopher Nolan direct?",
        expected_answers=["Inception", "The Dark Knight", "Interstellar"],
        question_type="aggregation",
    ),
    TestQuestion(
        question="What movies has Leonardo DiCaprio starred in?",
        expected_answers=["Inception", "Titanic"],
        question_type="aggregation",
    ),
]


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class ReasonerResult:
    """Result from a single reasoner on a single question."""
    reasoner_name: str
    question: str
    answer: str
    status: str
    confidence: float
    latency_ms: float
    llm_calls: int
    correct: bool


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    return answer.lower().strip().replace("_", " ")


def check_answer(predicted: str, expected: list[str]) -> bool:
    """Check if predicted answer matches any expected answer."""
    pred_norm = normalize_answer(predicted)
    for exp in expected:
        if normalize_answer(exp) in pred_norm or pred_norm in normalize_answer(exp):
            return True
    return False


def evaluate_reasoner(
    reasoner,
    reasoner_name: str,
    questions: list[TestQuestion],
    llm: OllamaBackend,
    verbose: bool = True,
) -> list[ReasonerResult]:
    """Evaluate a reasoner on all questions."""
    results = []
    
    for q in questions:
        llm.reset_count()
        start_time = time.time()
        
        try:
            result = reasoner.reason(q.question)
            latency_ms = (time.time() - start_time) * 1000
            
            is_correct = check_answer(result.answer, q.expected_answers)
            
            results.append(ReasonerResult(
                reasoner_name=reasoner_name,
                question=q.question,
                answer=result.answer[:50] if result.answer else "(no answer)",
                status=result.status.value if hasattr(result.status, 'value') else str(result.status),
                confidence=result.confidence,
                latency_ms=latency_ms,
                llm_calls=llm.call_count,
                correct=is_correct,
            ))
            
        except Exception as e:
            results.append(ReasonerResult(
                reasoner_name=reasoner_name,
                question=q.question,
                answer=f"ERROR: {str(e)[:30]}",
                status="error",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                llm_calls=llm.call_count,
                correct=False,
            ))
    
    return results


# =============================================================================
# Main
# =============================================================================

def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 70}")
    print(f"  {text}")
    print(f"{char * 70}")


def print_results_table(all_results: dict[str, list[ReasonerResult]]):
    """Print a comparison table of all results."""
    print_header("RESULTS COMPARISON", "=")
    
    # Summary by reasoner
    print("\n📊 Summary by Reasoner:")
    print("-" * 70)
    print(f"{'Reasoner':<25} {'Accuracy':>10} {'Avg Latency':>12} {'Avg LLM Calls':>15}")
    print("-" * 70)
    
    for name, results in all_results.items():
        accuracy = sum(r.correct for r in results) / len(results) if results else 0
        avg_latency = sum(r.latency_ms for r in results) / len(results) if results else 0
        avg_llm_calls = sum(r.llm_calls for r in results) / len(results) if results else 0
        
        print(f"{name:<25} {accuracy:>10.1%} {avg_latency:>10.0f}ms {avg_llm_calls:>15.1f}")
    
    # Detailed results by question
    print("\n\n📋 Detailed Results by Question:")
    print("-" * 70)
    
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\nQ{i+1} [{q.question_type}]: {q.question}")
        print(f"    Expected: {q.expected_answers[:2]}...")
        
        for name, results in all_results.items():
            r = results[i]
            status_icon = "✅" if r.correct else "❌"
            print(f"    {status_icon} {name:<20}: {r.answer:<30} ({r.latency_ms:.0f}ms)")


def main():
    parser = argparse.ArgumentParser(description="Compare reasoning paradigms with Ollama")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    args = parser.parse_args()
    
    print_header("NeuroSym-KG: Comparing Reasoning Paradigms", "=")
    print(f"Model: {args.model}")
    print(f"Ollama URL: {args.base_url}")
    
    # Test Ollama connection
    print("\n1. Testing Ollama connection...")
    try:
        test_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        response = test_llm.generate([Message(role="user", content="Say OK")])
        print(f"   ✅ Connected! Response: {response.content.strip()[:20]}")
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return
    
    # Create knowledge graph
    print("\n2. Creating Knowledge Graph...")
    kg = create_movie_kg()
    print(f"   ✅ Created: {kg.num_entities} entities, {kg.num_triples} triples")
    
    # Initialize all reasoners
    print("\n3. Initializing Reasoners...")
    
    reasoners = {}
    
    # Think-on-Graph
    try:
        tog_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        reasoners["Think-on-Graph (ToG)"] = (
            ThinkOnGraphReasoner(kg=kg, llm=tog_llm, max_depth=2, beam_width=3, verbose=args.verbose),
            tog_llm
        )
        print("   ✅ Think-on-Graph initialized")
    except Exception as e:
        print(f"   ❌ Think-on-Graph failed: {e}")
    
    # Reasoning on Graphs
    try:
        rog_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        reasoners["Reasoning on Graphs (RoG)"] = (
            ReasoningOnGraphs(kg=kg, llm=rog_llm, max_path_length=3, num_plans=2, verbose=args.verbose),
            rog_llm
        )
        print("   ✅ Reasoning on Graphs initialized")
    except Exception as e:
        print(f"   ❌ Reasoning on Graphs failed: {e}")
    
    # GraphRAG
    try:
        grag_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        reasoners["GraphRAG"] = (
            GraphRAGReasoner(kg=kg, llm=grag_llm, verbose=args.verbose),
            grag_llm
        )
        print("   ✅ GraphRAG initialized")
    except Exception as e:
        print(f"   ❌ GraphRAG failed: {e}")
    
    # SubgraphRAG
    try:
        srag_llm = OllamaBackend(model=args.model, base_url=args.base_url)
        reasoners["SubgraphRAG"] = (
            SubgraphRAGReasoner(kg=kg, llm=srag_llm, subgraph_size=30, verbose=args.verbose),
            srag_llm
        )
        print("   ✅ SubgraphRAG initialized")
    except Exception as e:
        print(f"   ❌ SubgraphRAG failed: {e}")
    
    if not reasoners:
        print("\n❌ No reasoners could be initialized!")
        return
    
    # Run evaluation
    print_header("Running Evaluation", "-")
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"Reasoners: {len(reasoners)}")
    
    all_results = {}
    
    for name, (reasoner, llm) in reasoners.items():
        print(f"\n🔄 Evaluating: {name}")
        results = evaluate_reasoner(reasoner, name, TEST_QUESTIONS, llm, verbose=args.verbose)
        all_results[name] = results
        
        accuracy = sum(r.correct for r in results) / len(results)
        print(f"   Accuracy: {accuracy:.1%}")
    
    # Print comparison
    print_results_table(all_results)
    
    # Print ASCII bar chart
    print_header("Accuracy Comparison", "-")
    max_bar = 40
    for name, results in all_results.items():
        accuracy = sum(r.correct for r in results) / len(results) if results else 0
        bar_len = int(accuracy * max_bar)
        bar = "█" * bar_len + "░" * (max_bar - bar_len)
        print(f"{name:<25}")
        print(f"  [{bar}] {accuracy:.1%}\n")
    
    # Recommendations
    print_header("Recommendations", "-")
    print("""
Based on the results:

• For simple 1-hop queries: SubgraphRAG is often fastest
• For multi-hop reasoning: ToG or RoG provide better path exploration  
• For aggregation queries: GraphRAG excels with community-based retrieval
• For explainability: RoG provides faithful reasoning paths

Tips:
• Try a larger model (qwen2.5-coder:7b) for better accuracy
• Increase beam_width/num_plans for more thorough search
• Check the verbose output to debug entity extraction issues
""")


if __name__ == "__main__":
    main()
