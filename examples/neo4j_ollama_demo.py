#!/usr/bin/env python3
"""
Demo: Neo4j Knowledge Graph + Ollama LLM Reasoning.

Combines:
- Neo4jKG: Production graph database for storage
- InMemoryKG: Fast graph for reasoning
- OllamaBackend: Local LLM inference
- ThinkOnGraphReasoner: LLM-guided graph traversal

This demonstrates a common pattern: use a production database for storage
and export to an in-memory graph for fast reasoning.

Prerequisites:
    1. Neo4j running (Neo4j Desktop or Docker)
    2. Ollama running: ollama serve
    3. Model pulled: ollama pull qwen2.5-coder:7b

Usage:
    python examples/neo4j_ollama_demo.py
    python examples/neo4j_ollama_demo.py --model llama3.2
"""

import argparse

from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.knowledge_graphs import Neo4jKG
from neurosym_kg.llm_backends import OllamaBackend
from neurosym_kg.core.types import Message
from neurosym_kg.reasoners import ThinkOnGraphReasoner


def get_scientist_triples():
    """Get scientist knowledge graph triples."""
    return [
        # Albert Einstein
        Triple("Albert_Einstein", "type", "Scientist"),
        Triple("Albert_Einstein", "born_in", "Ulm"),
        Triple("Albert_Einstein", "birth_year", "1879"),
        Triple("Albert_Einstein", "field", "Physics"),
        Triple("Albert_Einstein", "nationality", "German"),
        Triple("Albert_Einstein", "award", "Nobel_Prize_Physics_1921"),
        Triple("Albert_Einstein", "known_for", "Theory_of_Relativity"),
        Triple("Albert_Einstein", "known_for", "E_equals_mc_squared"),
        Triple("Albert_Einstein", "worked_at", "Princeton_University"),
        Triple("Albert_Einstein", "worked_at", "ETH_Zurich"),
        
        # Marie Curie
        Triple("Marie_Curie", "type", "Scientist"),
        Triple("Marie_Curie", "born_in", "Warsaw"),
        Triple("Marie_Curie", "birth_year", "1867"),
        Triple("Marie_Curie", "field", "Physics"),
        Triple("Marie_Curie", "field", "Chemistry"),
        Triple("Marie_Curie", "nationality", "Polish"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Physics_1903"),
        Triple("Marie_Curie", "award", "Nobel_Prize_Chemistry_1911"),
        Triple("Marie_Curie", "known_for", "Radioactivity"),
        Triple("Marie_Curie", "known_for", "Polonium"),
        Triple("Marie_Curie", "known_for", "Radium"),
        
        # Richard Feynman
        Triple("Richard_Feynman", "type", "Scientist"),
        Triple("Richard_Feynman", "born_in", "New_York_City"),
        Triple("Richard_Feynman", "birth_year", "1918"),
        Triple("Richard_Feynman", "field", "Physics"),
        Triple("Richard_Feynman", "nationality", "American"),
        Triple("Richard_Feynman", "award", "Nobel_Prize_Physics_1965"),
        Triple("Richard_Feynman", "known_for", "Quantum_Electrodynamics"),
        Triple("Richard_Feynman", "known_for", "Feynman_Diagrams"),
        Triple("Richard_Feynman", "worked_at", "Caltech"),
        Triple("Richard_Feynman", "worked_at", "Cornell_University"),
        
        # Isaac Newton
        Triple("Isaac_Newton", "type", "Scientist"),
        Triple("Isaac_Newton", "born_in", "Woolsthorpe"),
        Triple("Isaac_Newton", "birth_year", "1643"),
        Triple("Isaac_Newton", "field", "Physics"),
        Triple("Isaac_Newton", "field", "Mathematics"),
        Triple("Isaac_Newton", "nationality", "English"),
        Triple("Isaac_Newton", "known_for", "Laws_of_Motion"),
        Triple("Isaac_Newton", "known_for", "Calculus"),
        Triple("Isaac_Newton", "known_for", "Universal_Gravitation"),
        Triple("Isaac_Newton", "worked_at", "Cambridge_University"),
        
        # Locations
        Triple("Ulm", "located_in", "Germany"),
        Triple("Ulm", "type", "City"),
        Triple("Warsaw", "located_in", "Poland"),
        Triple("Warsaw", "type", "City"),
        Triple("New_York_City", "located_in", "United_States"),
        Triple("New_York_City", "type", "City"),
        Triple("Woolsthorpe", "located_in", "England"),
        Triple("Woolsthorpe", "type", "Village"),
        
        # Countries
        Triple("Germany", "type", "Country"),
        Triple("Poland", "type", "Country"),
        Triple("United_States", "type", "Country"),
        Triple("England", "type", "Country"),
        
        # Universities
        Triple("Princeton_University", "type", "University"),
        Triple("Princeton_University", "located_in", "United_States"),
        Triple("Caltech", "type", "University"),
        Triple("Caltech", "located_in", "United_States"),
        Triple("Cambridge_University", "type", "University"),
        Triple("Cambridge_University", "located_in", "England"),
        Triple("ETH_Zurich", "type", "University"),
        Triple("ETH_Zurich", "located_in", "Switzerland"),
        Triple("Cornell_University", "type", "University"),
        Triple("Cornell_University", "located_in", "United_States"),
        
        # Awards
        Triple("Nobel_Prize_Physics_1921", "type", "Award"),
        Triple("Nobel_Prize_Physics_1921", "year", "1921"),
        Triple("Nobel_Prize_Physics_1903", "type", "Award"),
        Triple("Nobel_Prize_Physics_1903", "year", "1903"),
        Triple("Nobel_Prize_Chemistry_1911", "type", "Award"),
        Triple("Nobel_Prize_Chemistry_1911", "year", "1911"),
        Triple("Nobel_Prize_Physics_1965", "type", "Award"),
        Triple("Nobel_Prize_Physics_1965", "year", "1965"),
        
        # Relationships between scientists
        Triple("Richard_Feynman", "influenced_by", "Albert_Einstein"),
        Triple("Albert_Einstein", "influenced_by", "Isaac_Newton"),
    ]


def main():
    parser = argparse.ArgumentParser(description="Neo4j + Ollama Demo")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Ollama model")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username")
    parser.add_argument("--neo4j-pass", default="password", help="Neo4j password")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Neo4j + Ollama Reasoning Demo")
    print("=" * 60)
    
    triples = get_scientist_triples()
    
    # =========================================================================
    # Part 1: Neo4j as Production Database
    # =========================================================================
    print("\n" + "─" * 60)
    print("PART 1: Neo4j as Production Database")
    print("─" * 60)
    
    print("\n1. Connecting to Neo4j...")
    try:
        neo4j_kg = Neo4jKG(
            uri=args.neo4j_uri,
            username=args.neo4j_user,
            password=args.neo4j_pass,
        )
        print(f"   ✓ Connected to Neo4j at {args.neo4j_uri}")
    except Exception as e:
        print(f"   ✗ Failed to connect to Neo4j: {e}")
        print("   Make sure Neo4j is running in Neo4j Desktop")
        print("\n   Continuing with InMemoryKG only...")
        neo4j_kg = None
    
    if neo4j_kg:
        # Populate Neo4j
        print("\n2. Populating Neo4j...")
        neo4j_kg.clear()
        neo4j_kg.add_triples(triples)
        print(f"   ✓ Neo4j: {neo4j_kg.num_entities} entities, {neo4j_kg.num_triples} triples")
        
        # Demonstrate Neo4j queries
        print("\n3. Neo4j Query Examples:")
        
        # Find Nobel Prize winners using Cypher
        winners = neo4j_kg.execute_cypher("""
            MATCH (scientist)-[:award]->(prize)
            WHERE prize.id CONTAINS 'Nobel'
            RETURN scientist.id as scientist, prize.id as prize
        """)
        print(f"   Nobel Prize Winners:")
        for w in winners:
            print(f"     • {w['scientist']}: {w['prize']}")
        
        # Find shortest path
        path = neo4j_kg.get_shortest_path("Richard_Feynman", "Isaac_Newton")
        if path:
            print(f"\n   Path from Feynman to Newton ({len(path)} hops):")
            for t in path:
                print(f"     • {t.subject} --{t.predicate}--> {t.object}")
        
        # Close Neo4j
        neo4j_kg.close()
        print("\n   ✓ Neo4j connection closed")
    
    # =========================================================================
    # Part 2: InMemoryKG for Reasoning
    # =========================================================================
    print("\n" + "─" * 60)
    print("PART 2: InMemoryKG for Fast Reasoning")
    print("─" * 60)
    
    print("\n4. Creating InMemoryKG for reasoning...")
    memory_kg = InMemoryKG(name="Scientists KG")
    memory_kg.add_triples(triples)
    print(f"   ✓ InMemoryKG: {memory_kg.num_entities} entities, {memory_kg.num_triples} triples")
    
    # =========================================================================
    # Part 3: Ollama LLM
    # =========================================================================
    print("\n" + "─" * 60)
    print("PART 3: Ollama Local LLM")
    print("─" * 60)
    
    print(f"\n5. Connecting to Ollama ({args.model})...")
    try:
        llm = OllamaBackend(model=args.model, temperature=0.1)
        test = llm.generate([Message(role="user", content="Say OK")])
        print(f"   ✓ Ollama connected! Test: {test.content.strip()[:20]}")
    except Exception as e:
        print(f"   ✗ Failed to connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return
    
    # =========================================================================
    # Part 4: Reasoning
    # =========================================================================
    print("\n" + "─" * 60)
    print("PART 4: ThinkOnGraph Reasoning")
    print("─" * 60)
    
    print("\n6. Creating ThinkOnGraph Reasoner...")
    reasoner = ThinkOnGraphReasoner(
        kg=memory_kg,  # Use InMemoryKG for reasoning
        llm=llm,
        max_depth=3,
        beam_width=3,
        verbose=True,
    )
    print("   ✓ Reasoner initialized")
    
    # Test questions
    questions = [
        "Where was Einstein born?",
        "What country was Marie Curie born in?",
        "What is Feynman known for?",
        "Who influenced Richard Feynman?",
        "What field did Newton work in?",
        "Where did Einstein work?",
    ]
    
    print("\n" + "=" * 60)
    print("Question Answering")
    print("=" * 60)
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 60}")
        print(f"Q{i}: {question}")
        print("─" * 60)
        
        try:
            result = reasoner.reason(question)
            
            print(f"Answer: {result.answer}")
            print(f"Status: {result.status}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Latency: {result.latency_ms:.0f}ms")
            
            if result.paths:
                print(f"Evidence paths: {len(result.paths)}")
                for path in result.paths[:2]:
                    path_str = " → ".join([
                        f"{t.subject}--{t.predicate}-->{t.object}" 
                        for t in path
                    ])
                    print(f"  • {path_str[:80]}...")
            
            results.append({
                "question": question,
                "answer": result.answer,
                "status": str(result.status),
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"question": question, "error": str(e)})
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    successful = [r for r in results if "error" not in r and r.get("answer")]
    print(f"Questions answered: {len(successful)}/{len(questions)}")
    
    if successful:
        avg_confidence = sum(r["confidence"] for r in successful) / len(successful)
        avg_latency = sum(r["latency_ms"] for r in successful) / len(successful)
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Average latency: {avg_latency:.0f}ms")
    
    # Stats
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    print(f"InMemoryKG entities: {memory_kg.num_entities}")
    print(f"InMemoryKG triples: {memory_kg.num_triples}")
    print(f"Reasoner stats: {reasoner.stats}")
    print(f"LLM stats: {llm.stats}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
