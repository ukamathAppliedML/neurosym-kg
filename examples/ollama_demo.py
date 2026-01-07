#!/usr/bin/env python3
"""
Demo script using Ollama as the LLM backend for NeuroSym-KG.

Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull llama3.2

Usage:
    python examples/ollama_demo.py
"""

from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.llm_backends import OllamaBackend
from neurosym_kg.core.types import Message
from neurosym_kg.reasoners import ThinkOnGraphReasoner


def create_scientist_kg() -> InMemoryKG:
    """Create a knowledge graph about scientists."""
    kg = InMemoryKG(name="Scientists KG")
    
    kg.add_triples([
        # Albert Einstein
        Triple("Albert_Einstein", "born_in", "Ulm"),
        Triple("Albert_Einstein", "birth_year", "1879"),
        Triple("Albert_Einstein", "field", "Physics"),
        Triple("Albert_Einstein", "nationality", "German"),
        Triple("Albert_Einstein", "award", "Nobel_Prize_Physics_1921"),
        Triple("Albert_Einstein", "known_for", "Theory_of_Relativity"),
        Triple("Albert_Einstein", "worked_at", "Princeton_University"),
        
        # Marie Curie
        Triple("Marie_Curie", "born_in", "Warsaw"),
        Triple("Marie_Curie", "birth_year", "1867"),
        Triple("Marie_Curie", "field", "Physics"),
        Triple("Marie_Curie", "field", "Chemistry"),
        Triple("Marie_Curie", "nationality", "Polish"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Physics_1903"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Chemistry_1911"),
        Triple("Marie_Curie", "known_for", "Radioactivity"),
        
        # Isaac Newton
        Triple("Isaac_Newton", "born_in", "Woolsthorpe"),
        Triple("Isaac_Newton", "birth_year", "1643"),
        Triple("Isaac_Newton", "field", "Physics"),
        Triple("Isaac_Newton", "field", "Mathematics"),
        Triple("Isaac_Newton", "nationality", "English"),
        Triple("Isaac_Newton", "known_for", "Laws_of_Motion"),
        Triple("Isaac_Newton", "known_for", "Calculus"),
        
        # Locations
        Triple("Ulm", "located_in", "Germany"),
        Triple("Ulm", "type", "City"),
        Triple("Warsaw", "located_in", "Poland"),
        Triple("Warsaw", "type", "City"),
        Triple("Woolsthorpe", "located_in", "England"),
        Triple("Germany", "type", "Country"),
        Triple("Poland", "type", "Country"),
        Triple("England", "type", "Country"),
        
        # Awards
        Triple("Nobel_Prize_Physics_1921", "type", "Award"),
        Triple("Nobel_Prize_Physics_1903", "type", "Award"),
        Triple("Nobel_Prize_Chemistry_1911", "type", "Award"),
    ])
    
    return kg


def main():
    print("=" * 60)
    print("NeuroSym-KG with Ollama Backend")
    print("=" * 60)
    
    # Create knowledge graph
    print("\n1. Creating Knowledge Graph...")
    kg = create_scientist_kg()
    print(kg.summary())
    
    # Create Ollama backend
    print("\n2. Connecting to Ollama...")
    try:
        llm = OllamaBackend(model="llama3.2")
        
        # Test connection
        test_response = llm.generate([
            Message(role="user", content="Reply with just: OK")
        ])
        print(f"   Ollama connected! Test response: {test_response.content.strip()[:20]}")
    except Exception as e:
        print(f"   Error connecting to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return
    
    # Create reasoner
    print("\n3. Initializing Think-on-Graph Reasoner...")
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
        "What field did Marie Curie work in?",
        "What is Einstein known for?",
    ]
    
    print("\n" + "=" * 60)
    print("Question Answering")
    print("=" * 60)
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        try:
            result = reasoner.reason(question)
            
            print(f"✅ Answer: {result.answer}")
            print(f"   Status: {result.status}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Latency: {result.latency_ms:.0f}ms")
            
            if result.paths:
                print(f"   Paths found: {len(result.paths)}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Print stats
    print("\n" + "=" * 60)
    print("Reasoner Statistics")
    print("=" * 60)
    stats = reasoner.stats
    print(f"  Queries: {stats.get('queries', 0)}")
    print(f"  KG calls: {stats.get('kg_calls', 0)}")
    print(f"  LLM calls: {stats.get('llm_calls', 0)}")


if __name__ == "__main__":
    main()
