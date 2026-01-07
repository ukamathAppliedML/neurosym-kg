#!/usr/bin/env python3
"""
Comparing Reasoners Example

Demonstrates how to compare different reasoning paradigms
on the same knowledge graph and questions.
"""

from neurosym_kg import (
    InMemoryKG,
    Triple,
    MockLLMBackend,
)
from neurosym_kg.reasoners import (
    ThinkOnGraphReasoner,
    ReasoningOnGraphs,
    SubgraphRAGReasoner,
)
from neurosym_kg.evaluation import (
    MetricsCalculator,
    exact_match,
    f1_score,
)


def create_movie_kg() -> InMemoryKG:
    """Create a sample movie knowledge graph."""
    kg = InMemoryKG(name="Movies KG")
    
    kg.add_triples([
        # Inception
        Triple("Inception", "director", "Christopher_Nolan"),
        Triple("Inception", "year", "2010"),
        Triple("Inception", "genre", "Science_Fiction"),
        Triple("Inception", "starring", "Leonardo_DiCaprio"),
        Triple("Inception", "starring", "Tom_Hardy"),
        
        # Christopher Nolan
        Triple("Christopher_Nolan", "born_in", "London"),
        Triple("Christopher_Nolan", "nationality", "British"),
        Triple("Christopher_Nolan", "occupation", "Director"),
        
        # The Dark Knight
        Triple("The_Dark_Knight", "director", "Christopher_Nolan"),
        Triple("The_Dark_Knight", "year", "2008"),
        Triple("The_Dark_Knight", "genre", "Action"),
        Triple("The_Dark_Knight", "starring", "Christian_Bale"),
        Triple("The_Dark_Knight", "starring", "Heath_Ledger"),
        
        # Leonardo DiCaprio
        Triple("Leonardo_DiCaprio", "born_in", "Los_Angeles"),
        Triple("Leonardo_DiCaprio", "occupation", "Actor"),
        
        # Titanic
        Triple("Titanic", "director", "James_Cameron"),
        Triple("Titanic", "year", "1997"),
        Triple("Titanic", "starring", "Leonardo_DiCaprio"),
        Triple("Titanic", "starring", "Kate_Winslet"),
        
        # Locations
        Triple("London", "located_in", "United_Kingdom"),
        Triple("Los_Angeles", "located_in", "California"),
        Triple("California", "located_in", "United_States"),
    ])
    
    return kg


def create_mock_llm() -> MockLLMBackend:
    """Create a mock LLM with responses for movie questions."""
    llm = MockLLMBackend()
    
    # Entity extraction patterns
    llm.add_response(r".*Extract.*Inception.*", "Inception")
    llm.add_response(r".*Extract.*Dark Knight.*", "The_Dark_Knight")
    llm.add_response(r".*Extract.*Nolan.*", "Christopher_Nolan")
    llm.add_response(r".*Extract.*DiCaprio.*", "Leonardo_DiCaprio")
    
    # Relation selection
    llm.add_response(r".*relations.*director.*", "director")
    llm.add_response(r".*relations.*born.*", "born_in")
    llm.add_response(r".*relations.*star.*", "starring")
    
    # Plan generation for RoG
    llm.add_response(
        r".*relation paths.*director.*",
        "director"
    )
    llm.add_response(
        r".*relation paths.*born.*",
        "director -> born_in"
    )
    
    # Reasoning checks
    llm.add_response(r".*enough information.*", "YES")
    
    # Answer generation
    llm.add_response(
        r".*answer.*director.*Inception.*",
        "Christopher Nolan directed Inception."
    )
    llm.add_response(
        r".*answer.*born.*director.*Inception.*",
        "Christopher Nolan was born in London."
    )
    llm.add_response(
        r".*answer.*movies.*Nolan.*",
        "Christopher Nolan directed Inception and The Dark Knight."
    )
    
    return llm


def main():
    print("=" * 70)
    print("Comparing Reasoning Paradigms")
    print("=" * 70)
    
    # Set up knowledge graph and LLM
    kg = create_movie_kg()
    print(f"\n{kg.summary()}\n")
    
    # Test questions with ground truth
    test_cases = [
        {
            "question": "Who directed Inception?",
            "answers": ["Christopher Nolan", "Christopher_Nolan"],
        },
        {
            "question": "Where was the director of Inception born?",
            "answers": ["London"],
        },
        {
            "question": "What movies did Christopher Nolan direct?",
            "answers": ["Inception", "The Dark Knight"],
        },
    ]
    
    # Create reasoners
    reasoners = {
        "Think-on-Graph": ThinkOnGraphReasoner(
            kg=kg,
            llm=create_mock_llm(),
            max_depth=2,
            beam_width=3,
        ),
        "Reasoning on Graphs": ReasoningOnGraphs(
            kg=kg,
            llm=create_mock_llm(),
            max_path_length=3,
            num_plans=3,
        ),
        "SubgraphRAG": SubgraphRAGReasoner(
            kg=kg,
            llm=create_mock_llm(),
            subgraph_size=20,
        ),
    }
    
    # Evaluate each reasoner
    results = {}
    
    for name, reasoner in reasoners.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {name}")
        print("=" * 70)
        
        calculator = MetricsCalculator(dataset_name=name)
        
        for tc in test_cases:
            question = tc["question"]
            ground_truth = tc["answers"]
            
            print(f"\nQ: {question}")
            
            result = reasoner.reason(question)
            
            print(f"A: {result.answer}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Latency: {result.latency_ms:.0f}ms")
            
            # Calculate metrics
            em = exact_match(result.answer, ground_truth)
            f1 = f1_score(result.answer, ground_truth)
            print(f"   EM: {em:.2f}, F1: {f1:.2f}")
            
            calculator.add_prediction(
                prediction=result.answer,
                ground_truth=ground_truth,
                question=question,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
            )
        
        results[name] = calculator.compute()
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"\n{'Reasoner':<25} {'Accuracy':>10} {'F1':>10} {'Avg Latency':>12}")
    print("-" * 60)
    
    for name, eval_result in results.items():
        print(
            f"{name:<25} "
            f"{eval_result.accuracy:>10.2%} "
            f"{eval_result.f1:>10.2%} "
            f"{eval_result.metrics['avg_latency_ms']:>10.0f}ms"
        )
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
