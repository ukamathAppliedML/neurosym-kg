#!/usr/bin/env python3
"""
Basic Question Answering Example

Demonstrates how to use NeuroSym-KG for simple knowledge graph QA.
"""

from neurosym_kg import (
    InMemoryKG,
    Triple,
    ThinkOnGraphReasoner,
    MockLLMBackend,
)


def main():
    # Create a simple knowledge graph about scientists
    kg = InMemoryKG(name="Scientists KG")
    
    kg.add_triples([
        # Albert Einstein
        Triple("Albert_Einstein", "born_in", "Ulm"),
        Triple("Albert_Einstein", "field", "Physics"),
        Triple("Albert_Einstein", "nationality", "German"),
        Triple("Albert_Einstein", "award", "Nobel_Prize_Physics_1921"),
        Triple("Albert_Einstein", "known_for", "Theory_of_Relativity"),
        Triple("Albert_Einstein", "worked_at", "Princeton_University"),
        
        # Ulm
        Triple("Ulm", "located_in", "Germany"),
        Triple("Ulm", "type", "City"),
        
        # Marie Curie
        Triple("Marie_Curie", "born_in", "Warsaw"),
        Triple("Marie_Curie", "field", "Physics"),
        Triple("Marie_Curie", "field", "Chemistry"),
        Triple("Marie_Curie", "nationality", "Polish"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Physics_1903"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Chemistry_1911"),
        Triple("Marie_Curie", "known_for", "Radioactivity"),
        
        # Warsaw
        Triple("Warsaw", "located_in", "Poland"),
        Triple("Warsaw", "type", "City"),
        
        # Isaac Newton
        Triple("Isaac_Newton", "born_in", "Woolsthorpe"),
        Triple("Isaac_Newton", "field", "Physics"),
        Triple("Isaac_Newton", "field", "Mathematics"),
        Triple("Isaac_Newton", "nationality", "English"),
        Triple("Isaac_Newton", "known_for", "Laws_of_Motion"),
        Triple("Isaac_Newton", "known_for", "Calculus"),
        
        # Woolsthorpe
        Triple("Woolsthorpe", "located_in", "England"),
        Triple("England", "part_of", "United_Kingdom"),
    ])
    
    print(kg.summary())
    print()
    
    # Create a mock LLM backend with predefined responses
    llm = MockLLMBackend(default_response="Unknown")
    
    # Add some pattern-based responses for entity extraction
    llm.add_response(
        r".*Extract.*entities.*Einstein.*",
        "Albert_Einstein"
    )
    llm.add_response(
        r".*Extract.*entities.*Curie.*",
        "Marie_Curie"
    )
    llm.add_response(
        r".*Extract.*entities.*born.*",
        "Albert_Einstein"
    )
    
    # Add responses for relation selection
    llm.add_response(
        r".*relevant relations.*born.*",
        "born_in"
    )
    llm.add_response(
        r".*relevant relations.*field.*",
        "field"
    )
    
    # Add reasoning check responses
    llm.add_response(
        r".*enough information.*",
        "YES - The knowledge graph contains the required information."
    )
    
    # Add answer generation responses
    llm.add_response(
        r".*answer.*born.*Einstein.*",
        "Albert Einstein was born in Ulm."
    )
    llm.add_response(
        r".*answer.*field.*",
        "Physics"
    )
    
    # Create the reasoner
    reasoner = ThinkOnGraphReasoner(
        kg=kg,
        llm=llm,
        max_depth=2,
        beam_width=3,
        verbose=True,
    )
    
    # Test questions
    questions = [
        "Where was Einstein born?",
        "What field did Einstein work in?",
    ]
    
    print("=" * 60)
    print("Question Answering Demo")
    print("=" * 60)
    
    for question in questions:
        print(f"\nQ: {question}")
        
        result = reasoner.reason(question)
        
        print(f"A: {result.answer}")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Latency: {result.latency_ms:.0f}ms")
        
        if result.paths:
            print(f"   Reasoning paths: {len(result.paths)}")
            for i, path in enumerate(result.paths[:2]):
                print(f"     Path {i+1}: {path.to_text()}")
    
    print("\n" + "=" * 60)
    print("Reasoner Statistics")
    print("=" * 60)
    print(f"  Queries: {reasoner.stats['queries']}")
    print(f"  KG calls: {reasoner.stats['kg_calls']}")
    print(f"  LLM calls: {reasoner.stats['llm_calls']}")
    print(f"  Success rate: {reasoner.stats.get('success_rate', 0):.2%}")


if __name__ == "__main__":
    main()
