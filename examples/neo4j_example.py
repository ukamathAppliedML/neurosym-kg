#!/usr/bin/env python3
"""
Example: Using Neo4j Knowledge Graph Backend.

Prerequisites:
1. Install Neo4j: https://neo4j.com/download/
   Or use Docker: docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

2. Install Python driver:
   pip install neo4j

Usage:
    python neo4j_example.py
"""

from neurosym_kg import Triple
from neurosym_kg.knowledge_graphs import Neo4jKG
from neurosym_kg.core.types import Entity


def main():
    # Connect to Neo4j
    print("Connecting to Neo4j...")
    
    try:
        kg = Neo4jKG(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",  # Change to your password
            database="neo4j",
        )
        print(f"✓ Connected to Neo4j")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest")
        return
    
    # Clear existing data (optional)
    print("\nClearing existing data...")
    kg.clear()
    
    # Create indexes for performance
    print("Creating indexes...")
    kg.create_indexes()
    
    # Add some entities and triples
    print("\nAdding knowledge graph data...")
    
    triples = [
        # Scientists
        Triple("Albert_Einstein", "type", "Scientist"),
        Triple("Albert_Einstein", "born_in", "Germany"),
        Triple("Albert_Einstein", "worked_at", "Princeton"),
        Triple("Albert_Einstein", "developed", "Theory_of_Relativity"),
        Triple("Albert_Einstein", "won", "Nobel_Prize_Physics_1921"),
        
        Triple("Marie_Curie", "type", "Scientist"),
        Triple("Marie_Curie", "born_in", "Poland"),
        Triple("Marie_Curie", "worked_at", "University_of_Paris"),
        Triple("Marie_Curie", "discovered", "Radium"),
        Triple("Marie_Curie", "won", "Nobel_Prize_Physics_1903"),
        Triple("Marie_Curie", "won", "Nobel_Prize_Chemistry_1911"),
        
        Triple("Richard_Feynman", "type", "Scientist"),
        Triple("Richard_Feynman", "born_in", "United_States"),
        Triple("Richard_Feynman", "worked_at", "Caltech"),
        Triple("Richard_Feynman", "contributed_to", "Quantum_Electrodynamics"),
        Triple("Richard_Feynman", "won", "Nobel_Prize_Physics_1965"),
        
        # Relationships between scientists
        Triple("Richard_Feynman", "influenced_by", "Albert_Einstein"),
        Triple("Albert_Einstein", "collaborated_with", "Marie_Curie"),
        
        # Institutions
        Triple("Princeton", "type", "University"),
        Triple("Princeton", "located_in", "United_States"),
        Triple("Caltech", "type", "University"),
        Triple("Caltech", "located_in", "United_States"),
    ]
    
    kg.add_triples(triples)
    print(f"✓ Added {len(triples)} triples")
    
    # Query the graph
    print("\n" + "="*60)
    print("QUERYING THE KNOWLEDGE GRAPH")
    print("="*60)
    
    # 1. Get entity info
    print("\n1. Get entity: Albert_Einstein")
    entity = kg.get_entity("Albert_Einstein")
    if entity:
        print(f"   Found: {entity.id}")
    
    # 2. Get neighbors
    print("\n2. Get neighbors of Albert_Einstein:")
    neighbors = kg.get_neighbors("Albert_Einstein", limit=10)
    for n in neighbors:
        print(f"   - {n.id}")
    
    # 3. Get specific triples
    print("\n3. Find all scientists (type = Scientist):")
    scientists = kg.get_triples(predicate="type", object="Scientist")
    for t in scientists:
        print(f"   - {t.subject}")
    
    # 4. Get relations for an entity
    print("\n4. All relations for Marie_Curie:")
    relations = kg.get_relations("Marie_Curie")
    for r in relations:
        print(f"   - {r.name}: {r.description}")
    
    # 5. Find shortest path
    print("\n5. Shortest path from Richard_Feynman to Marie_Curie:")
    path = kg.get_shortest_path("Richard_Feynman", "Marie_Curie", max_depth=4)
    if path:
        for t in path:
            print(f"   {t.subject} --{t.predicate}--> {t.object}")
    else:
        print("   No path found")
    
    # 6. Search entities
    print("\n6. Search for 'Einstein':")
    results = kg.search_entities("Einstein", limit=5)
    for e in results:
        print(f"   - {e.id}: {e.name}")
    
    # 7. Execute custom Cypher
    print("\n7. Custom Cypher: Find all Nobel Prize winners")
    winners = kg.execute_cypher("""
        MATCH (person)-[:won]->(prize)
        WHERE prize.id CONTAINS 'Nobel'
        RETURN person.id as name, prize.id as prize
    """)
    for w in winners:
        print(f"   - {w['name']}: {w['prize']}")
    
    # 8. Get graph statistics
    print("\n8. Graph statistics:")
    print(f"   Entities: {kg.num_entities}")
    print(f"   Triples: {kg.num_triples}")
    print(f"   Relation types: {kg.get_relation_types()}")
    
    # Cleanup
    kg.close()
    print("\n✓ Connection closed")


if __name__ == "__main__":
    main()
