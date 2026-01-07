"""
In-memory Knowledge Graph implementation.

Provides a fast, lightweight KG that stores all data in memory.
Ideal for testing, prototyping, and small datasets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from neurosym_kg.core.types import Entity, EntityType, Relation, RelationType, Subgraph, Triple
from neurosym_kg.knowledge_graphs.base import BaseMutableKnowledgeGraph


class InMemoryKG(BaseMutableKnowledgeGraph):
    """
    In-memory Knowledge Graph implementation using dictionaries.

    Features:
    - Fast lookups via indexed data structures
    - Support for entity and relation metadata
    - Automatic ID normalization
    - Simple and lightweight

    Example:
        >>> kg = InMemoryKG()
        >>> kg.add_triple(Triple("Einstein", "born_in", "Ulm"))
        >>> kg.add_triple(Triple("Einstein", "field", "Physics"))
        >>> kg.get_neighbors("Einstein")
        [Triple(subject='Einstein', predicate='born_in', object='Ulm'), ...]
    """

    def __init__(self, name: str = "InMemoryKG") -> None:
        super().__init__(name)

        # Core storage
        self._entities: dict[str, Entity] = {}
        self._relations: dict[str, Relation] = {}
        self._triples: set[tuple[str, str, str]] = set()

        # Indexes for fast lookups
        self._subject_index: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self._object_index: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self._predicate_index: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

        # Name to ID mapping for entity lookup
        self._name_to_ids: dict[str, set[str]] = defaultdict(set)

    @property
    def num_entities(self) -> int:
        """Number of unique entities."""
        return len(self._entities)

    @property
    def num_triples(self) -> int:
        """Number of triples."""
        return len(self._triples)

    @property
    def num_relations(self) -> int:
        """Number of unique relations."""
        return len(self._relations)

    def _normalize_id(self, id_str: str) -> str:
        """Normalize an ID for consistent lookups."""
        return id_str.strip()

    def _ensure_entity(self, entity_id: str) -> Entity:
        """Ensure an entity exists, creating a basic one if not."""
        entity_id = self._normalize_id(entity_id)
        if entity_id not in self._entities:
            entity = Entity(id=entity_id, name=entity_id)
            self._entities[entity_id] = entity
            self._name_to_ids[entity_id.lower()].add(entity_id)
        return self._entities[entity_id]

    def _ensure_relation(self, relation_id: str) -> Relation:
        """Ensure a relation exists, creating a basic one if not."""
        relation_id = self._normalize_id(relation_id)
        if relation_id not in self._relations:
            relation = Relation(id=relation_id, name=relation_id)
            self._relations[relation_id] = relation
        return self._relations[relation_id]

    def _triple_to_tuple(self, triple: Triple) -> tuple[str, str, str]:
        """Convert Triple to normalized tuple."""
        return (
            self._normalize_id(triple.subject_id),
            self._normalize_id(triple.predicate_id),
            self._normalize_id(triple.object_id),
        )

    def _tuple_to_triple(self, t: tuple[str, str, str]) -> Triple:
        """Convert tuple back to Triple with full entity/relation info."""
        subj, pred, obj = t
        return Triple(
            subject=self._entities.get(subj, Entity(id=subj, name=subj)),
            predicate=self._relations.get(pred, Relation(id=pred, name=pred)),
            object=self._entities.get(obj, Entity(id=obj, name=obj)),
        )

    # Read operations

    def get_entity(self, entity_id: str) -> Entity | None:
        """Retrieve an entity by its ID."""
        self._increment_stat("queries")
        return self._entities.get(self._normalize_id(entity_id))

    def get_entity_by_name(self, name: str, limit: int = 10) -> list[Entity]:
        """Search for entities by name (case-insensitive)."""
        self._increment_stat("queries")
        results: list[Entity] = []
        name_lower = name.lower()

        # Exact match first
        if name_lower in self._name_to_ids:
            for eid in self._name_to_ids[name_lower]:
                if eid in self._entities:
                    results.append(self._entities[eid])

        # Partial match
        if len(results) < limit:
            for entity in self._entities.values():
                if entity not in results:
                    if name_lower in entity.name.lower():
                        results.append(entity)
                    elif any(name_lower in alias.lower() for alias in entity.aliases):
                        results.append(entity)
                if len(results) >= limit:
                    break

        return results[:limit]

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "both",
        relation_filter: list[str] | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Get neighboring triples for an entity."""
        self._increment_stat("queries")
        entity_id = self._normalize_id(entity_id)
        results: list[Triple] = []

        # Normalize relation filter
        rel_filter_set: set[str] | None = None
        if relation_filter:
            rel_filter_set = {self._normalize_id(r) for r in relation_filter}

        def matches_filter(pred: str) -> bool:
            return rel_filter_set is None or pred in rel_filter_set

        # Outgoing edges (entity is subject)
        if direction in ("outgoing", "both"):
            for t in self._subject_index.get(entity_id, set()):
                if len(results) >= limit:
                    break
                if matches_filter(t[1]):
                    results.append(self._tuple_to_triple(t))

        # Incoming edges (entity is object)
        if direction in ("incoming", "both") and len(results) < limit:
            for t in self._object_index.get(entity_id, set()):
                if len(results) >= limit:
                    break
                if matches_filter(t[1]):
                    results.append(self._tuple_to_triple(t))

        return results[:limit]

    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations connected to an entity."""
        self._increment_stat("queries")
        entity_id = self._normalize_id(entity_id)
        relation_ids: set[str] = set()

        # From subject index
        for t in self._subject_index.get(entity_id, set()):
            relation_ids.add(t[1])

        # From object index
        for t in self._object_index.get(entity_id, set()):
            relation_ids.add(t[1])

        return [self._relations[rid] for rid in relation_ids if rid in self._relations]

    def get_triples(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
        limit: int = 100,
    ) -> list[Triple]:
        """Query triples with optional filters."""
        self._increment_stat("queries")

        # Normalize filters
        subject = self._normalize_id(subject) if subject else None
        predicate = self._normalize_id(predicate) if predicate else None
        obj = self._normalize_id(obj) if obj else None

        # Choose the most selective index
        if subject:
            candidates = self._subject_index.get(subject, set())
        elif obj:
            candidates = self._object_index.get(obj, set())
        elif predicate:
            candidates = self._predicate_index.get(predicate, set())
        else:
            candidates = self._triples

        results: list[Triple] = []
        for t in candidates:
            if len(results) >= limit:
                break
            if subject and t[0] != subject:
                continue
            if predicate and t[1] != predicate:
                continue
            if obj and t[2] != obj:
                continue
            results.append(self._tuple_to_triple(t))

        return results

    # Write operations

    def add_entity(self, entity: Entity) -> bool:
        """Add an entity to the graph."""
        entity_id = self._normalize_id(entity.id)
        if entity_id in self._entities:
            # Update existing entity
            existing = self._entities[entity_id]
            # Merge properties
            existing.aliases = list(set(existing.aliases + entity.aliases))
            if entity.description and not existing.description:
                existing.description = entity.description
            existing.properties.update(entity.properties)
            return False

        self._entities[entity_id] = entity
        self._name_to_ids[entity.name.lower()].add(entity_id)
        for alias in entity.aliases:
            self._name_to_ids[alias.lower()].add(entity_id)
        return True

    def add_triple(self, triple: Triple) -> bool:
        """Add a triple to the graph."""
        t = self._triple_to_tuple(triple)

        if t in self._triples:
            return False

        # Ensure entities and relations exist
        self._ensure_entity(t[0])
        self._ensure_entity(t[2])
        self._ensure_relation(t[1])

        # Add to storage and indexes
        self._triples.add(t)
        self._subject_index[t[0]].add(t)
        self._object_index[t[2]].add(t)
        self._predicate_index[t[1]].add(t)

        return True

    def add_triples(self, triples: list[Triple] | Iterable[Triple]) -> int:
        """Add multiple triples efficiently."""
        count = 0
        for triple in triples:
            if self.add_triple(triple):
                count += 1
        return count

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its triples."""
        entity_id = self._normalize_id(entity_id)
        if entity_id not in self._entities:
            return False

        entity = self._entities[entity_id]

        # Remove all triples involving this entity
        triples_to_remove: list[tuple[str, str, str]] = []
        triples_to_remove.extend(self._subject_index.get(entity_id, set()))
        triples_to_remove.extend(self._object_index.get(entity_id, set()))

        for t in triples_to_remove:
            self._remove_triple_tuple(t)

        # Remove entity
        del self._entities[entity_id]
        self._name_to_ids[entity.name.lower()].discard(entity_id)
        for alias in entity.aliases:
            self._name_to_ids[alias.lower()].discard(entity_id)

        return True

    def remove_triple(self, triple: Triple) -> bool:
        """Remove a specific triple."""
        t = self._triple_to_tuple(triple)
        return self._remove_triple_tuple(t)

    def _remove_triple_tuple(self, t: tuple[str, str, str]) -> bool:
        """Remove a triple by tuple."""
        if t not in self._triples:
            return False

        self._triples.remove(t)
        self._subject_index[t[0]].discard(t)
        self._object_index[t[2]].discard(t)
        self._predicate_index[t[1]].discard(t)

        return True

    def clear(self) -> None:
        """Remove all entities and triples."""
        self._entities.clear()
        self._relations.clear()
        self._triples.clear()
        self._subject_index.clear()
        self._object_index.clear()
        self._predicate_index.clear()
        self._name_to_ids.clear()

    # Convenience methods

    @classmethod
    def from_triples(
        cls,
        triples: list[Triple] | list[tuple[str, str, str]],
        name: str = "InMemoryKG",
    ) -> "InMemoryKG":
        """Create a KG from a list of triples."""
        kg = cls(name=name)
        for t in triples:
            if isinstance(t, Triple):
                kg.add_triple(t)
            else:
                kg.add_triple(Triple(subject=t[0], predicate=t[1], object=t[2]))
        return kg

    def to_networkx(self) -> "networkx.MultiDiGraph":
        """Convert to NetworkX graph for analysis/visualization."""
        import networkx as nx

        G = nx.MultiDiGraph()

        # Add nodes with attributes
        for entity_id, entity in self._entities.items():
            G.add_node(
                entity_id,
                name=entity.name,
                entity_type=entity.entity_type.value,
                **entity.properties,
            )

        # Add edges
        for t in self._triples:
            G.add_edge(t[0], t[2], relation=t[1])

        return G

    def __repr__(self) -> str:
        return f"InMemoryKG(entities={self.num_entities}, triples={self.num_triples})"

    def summary(self) -> str:
        """Get a summary of the KG contents."""
        lines = [
            f"Knowledge Graph: {self.name}",
            f"  Entities: {self.num_entities}",
            f"  Relations: {self.num_relations}",
            f"  Triples: {self.num_triples}",
        ]

        # Entity type distribution
        type_counts: dict[str, int] = defaultdict(int)
        for entity in self._entities.values():
            type_counts[entity.entity_type.value] += 1

        if type_counts:
            lines.append("  Entity Types:")
            for etype, count in sorted(type_counts.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"    {etype}: {count}")

        # Top relations
        rel_counts: dict[str, int] = defaultdict(int)
        for t in self._triples:
            rel_counts[t[1]] += 1

        if rel_counts:
            lines.append("  Top Relations:")
            for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1])[:5]:
                rel_name = self._relations.get(rel, Relation(id=rel, name=rel)).name
                lines.append(f"    {rel_name}: {count}")

        return "\n".join(lines)
