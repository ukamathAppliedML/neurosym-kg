"""
Base class for Knowledge Graph implementations.

Provides common functionality and utilities that all KG backends can use.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from neurosym_kg.core.types import Entity, Relation, Subgraph, Triple


class BaseKnowledgeGraph(ABC):
    """
    Abstract base class for Knowledge Graph implementations.

    Provides common utilities and defines the interface that all
    KG backends must implement.
    """

    def __init__(self, name: str = "BaseKG") -> None:
        self._name = name
        self._stats: dict[str, Any] = {
            "queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    @property
    def name(self) -> str:
        """Human-readable name of the KG backend."""
        return self._name

    @property
    def stats(self) -> dict[str, Any]:
        """Query statistics."""
        return self._stats.copy()

    def _increment_stat(self, key: str, value: int = 1) -> None:
        """Increment a statistic counter."""
        self._stats[key] = self._stats.get(key, 0) + value

    @abstractmethod
    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by its ID."""
        ...

    @abstractmethod
    def get_entity_by_name(self, name: str, limit: int = 10) -> list[Entity]:
        """Search for entities by name."""
        ...

    @abstractmethod
    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_filter: list[str] | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Get neighboring triples for an entity."""
        ...

    @abstractmethod
    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations connected to an entity."""
        ...

    @abstractmethod
    def get_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Query triples with optional filters."""
        ...

    def get_subgraph(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
        max_triples: int = 100,
    ) -> Subgraph:
        """
        Extract a subgraph centered on given entities.

        Default implementation using BFS. Can be overridden for efficiency.
        """
        visited_entities: set[str] = set()
        collected_triples: list[Triple] = []

        # BFS from each seed entity
        current_frontier = set(entity_ids)

        for hop in range(max_hops):
            if len(collected_triples) >= max_triples:
                break

            next_frontier: set[str] = set()

            for entity_id in current_frontier:
                if entity_id in visited_entities:
                    continue
                visited_entities.add(entity_id)

                # Get neighbors
                neighbors = self.get_neighbors(
                    entity_id,
                    direction="both",
                    limit=max_triples - len(collected_triples),
                )

                for triple in neighbors:
                    if len(collected_triples) >= max_triples:
                        break
                    if triple not in collected_triples:
                        collected_triples.append(triple)
                        # Add new entities to explore
                        next_frontier.add(triple.subject_id)
                        next_frontier.add(triple.object_id)

            current_frontier = next_frontier - visited_entities

        return Subgraph(
            triples=collected_triples,
            entities=visited_entities,
            center_entity=entity_ids[0] if entity_ids else None,
        )

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> list[list[Triple]]:
        """
        Find paths between two entities using BFS.

        Default implementation. Can be overridden for efficiency.
        """
        if source == target:
            return [[]]

        # BFS to find paths
        # Each queue element: (current_entity, path_so_far)
        queue: list[tuple[str, list[Triple]]] = [(source, [])]
        found_paths: list[list[Triple]] = []
        visited_at_depth: dict[str, int] = {source: 0}

        while queue and len(found_paths) < max_paths:
            current, path = queue.pop(0)

            if len(path) >= max_hops:
                continue

            neighbors = self.get_neighbors(current, direction="outgoing", limit=50)

            for triple in neighbors:
                next_entity = triple.object_id
                new_path = path + [triple]

                # Found target
                if next_entity == target:
                    found_paths.append(new_path)
                    if len(found_paths) >= max_paths:
                        break
                    continue

                # Continue exploring if not too deep
                new_depth = len(new_path)
                if new_depth < max_hops:
                    # Allow revisiting at same or greater depth for different paths
                    if next_entity not in visited_at_depth or visited_at_depth[next_entity] >= new_depth:
                        visited_at_depth[next_entity] = new_depth
                        queue.append((next_entity, new_path))

        return found_paths

    def get_relation_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
    ) -> list[list[str]]:
        """
        Get unique relation sequences between two entities.

        Returns list of relation ID sequences, not full paths.
        """
        full_paths = self.find_paths(source, target, max_hops=max_hops)
        relation_paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for path in full_paths:
            relations = [t.predicate_id for t in path]
            rel_tuple = tuple(relations)
            if rel_tuple not in seen:
                seen.add(rel_tuple)
                relation_paths.append(relations)

        return relation_paths

    @staticmethod
    def _make_cache_key(*args: Any) -> str:
        """Create a cache key from arguments."""
        key_str = "|".join(str(a) for a in args)
        return hashlib.md5(key_str.encode()).hexdigest()


class BaseMutableKnowledgeGraph(BaseKnowledgeGraph):
    """
    Abstract base class for mutable Knowledge Graphs.

    Extends BaseKnowledgeGraph with write operations.
    """

    @abstractmethod
    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        ...

    @abstractmethod
    def add_triple(self, triple: Triple) -> bool:
        """Add a triple to the graph."""
        ...

    def add_triples(self, triples: list[Triple]) -> int:
        """Add multiple triples. Returns count of added triples."""
        count = 0
        for triple in triples:
            if self.add_triple(triple):
                count += 1
        return count

    @abstractmethod
    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its triples."""
        ...

    @abstractmethod
    def remove_triple(self, triple: Triple) -> bool:
        """Remove a specific triple."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all entities and triples."""
        ...
