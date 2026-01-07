"""
Reasoning engine implementations for NeuroSym-KG.

This module provides multiple reasoning paradigms:
- ThinkOnGraphReasoner: LLM as agent with beam search (ToG, ICLR 2024)
- ReasoningOnGraphs: Faithful plan-based reasoning (RoG, ICLR 2024)
- GraphRAGReasoner: Community-based retrieval (Microsoft 2024)
- SubgraphRAGReasoner: Flexible subgraph retrieval (2024)

Each reasoner follows the same interface, making it easy to swap
paradigms and compare performance.
"""

from neurosym_kg.reasoners.base import BaseReasoner
from neurosym_kg.reasoners.graph_rag import GraphRAGReasoner, LocalGraphRAG
from neurosym_kg.reasoners.reasoning_on_graphs import ReasoningOnGraphs, ReasoningPlan
from neurosym_kg.reasoners.subgraph_rag import HybridSubgraphRAG, SubgraphRAGReasoner
from neurosym_kg.reasoners.think_on_graph import ThinkOnGraphReasoner, ToGSearchState

__all__ = [
    "BaseReasoner",
    # Think-on-Graph
    "ThinkOnGraphReasoner",
    "ToGSearchState",
    # Reasoning on Graphs
    "ReasoningOnGraphs",
    "ReasoningPlan",
    # GraphRAG
    "GraphRAGReasoner",
    "LocalGraphRAG",
    # SubgraphRAG
    "SubgraphRAGReasoner",
    "HybridSubgraphRAG",
]
