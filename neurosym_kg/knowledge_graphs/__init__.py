"""
Knowledge Graph backends for NeuroSym-KG.

This module provides connectors for various knowledge graph sources:
- InMemoryKG: Fast in-memory graph for testing and prototyping
- WikidataKG: Wikidata via SPARQL
- Neo4jKG: Neo4j graph database (coming soon)
- FreebaseKG: Freebase dump reader (coming soon)
"""

from neurosym_kg.knowledge_graphs.base import BaseKnowledgeGraph, BaseMutableKnowledgeGraph
from neurosym_kg.knowledge_graphs.in_memory import InMemoryKG
from neurosym_kg.knowledge_graphs.wikidata import WikidataKG
from neurosym_kg.knowledge_graphs.neo4j_kg import Neo4jKG


__all__ = [
    "BaseKnowledgeGraph",
    "BaseMutableKnowledgeGraph",
    "InMemoryKG",
    "WikidataKG",
    "Neo4jKG",
]
