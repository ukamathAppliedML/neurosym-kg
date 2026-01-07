"""
Unit tests for InMemoryKG.
"""

import pytest
from neurosym_kg.knowledge_graphs import InMemoryKG
from neurosym_kg.core.types import Entity, EntityType, Triple


class TestInMemoryKG:
    """Tests for InMemoryKG class."""

    @pytest.fixture
    def empty_kg(self):
        """Create an empty KG."""
        return InMemoryKG(name="TestKG")

    @pytest.fixture
    def sample_kg(self):
        """Create a KG with sample data."""
        kg = InMemoryKG(name="SampleKG")
        kg.add_triples([
            Triple("Einstein", "born_in", "Ulm"),
            Triple("Einstein", "field", "Physics"),
            Triple("Ulm", "located_in", "Germany"),
            Triple("Curie", "born_in", "Warsaw"),
            Triple("Curie", "field", "Physics"),
            Triple("Curie", "field", "Chemistry"),
        ])
        return kg

    def test_empty_kg(self, empty_kg):
        """Test empty KG properties."""
        assert empty_kg.num_entities == 0
        assert empty_kg.num_triples == 0
        assert empty_kg.name == "TestKG"

    def test_add_triple(self, empty_kg):
        """Test adding a single triple."""
        result = empty_kg.add_triple(
            Triple("A", "rel", "B")
        )
        assert result is True
        assert empty_kg.num_triples == 1
        assert empty_kg.num_entities == 2

    def test_add_duplicate_triple(self, empty_kg):
        """Test adding duplicate triples."""
        triple = Triple("A", "rel", "B")
        empty_kg.add_triple(triple)
        result = empty_kg.add_triple(triple)

        assert result is False  # Duplicate not added
        assert empty_kg.num_triples == 1

    def test_add_triples_batch(self, empty_kg):
        """Test adding multiple triples."""
        triples = [
            Triple("A", "r1", "B"),
            Triple("B", "r2", "C"),
            Triple("A", "r1", "B"),  # Duplicate
        ]
        count = empty_kg.add_triples(triples)

        assert count == 2  # Only unique triples
        assert empty_kg.num_triples == 2

    def test_get_entity(self, sample_kg):
        """Test getting an entity by ID."""
        entity = sample_kg.get_entity("Einstein")
        assert entity is not None
        assert entity.id == "Einstein"

    def test_get_entity_not_found(self, sample_kg):
        """Test getting non-existent entity."""
        entity = sample_kg.get_entity("NonExistent")
        assert entity is None

    def test_get_entity_by_name(self, sample_kg):
        """Test searching entities by name."""
        entities = sample_kg.get_entity_by_name("Einstein")
        assert len(entities) >= 1
        assert any(e.id == "Einstein" for e in entities)

    def test_get_neighbors_outgoing(self, sample_kg):
        """Test getting outgoing neighbors."""
        neighbors = sample_kg.get_neighbors(
            "Einstein",
            direction="outgoing"
        )
        assert len(neighbors) == 2  # born_in and field
        assert all(t.subject_id == "Einstein" for t in neighbors)

    def test_get_neighbors_incoming(self, sample_kg):
        """Test getting incoming neighbors."""
        neighbors = sample_kg.get_neighbors(
            "Physics",
            direction="incoming"
        )
        # Both Einstein and Curie have field=Physics
        assert len(neighbors) == 2
        assert all(t.object_id == "Physics" for t in neighbors)

    def test_get_neighbors_with_filter(self, sample_kg):
        """Test filtering neighbors by relation."""
        neighbors = sample_kg.get_neighbors(
            "Einstein",
            direction="outgoing",
            relation_filter=["born_in"]
        )
        assert len(neighbors) == 1
        assert neighbors[0].predicate_id == "born_in"
        assert neighbors[0].object_id == "Ulm"

    def test_get_relations(self, sample_kg):
        """Test getting relations for an entity."""
        relations = sample_kg.get_relations("Einstein")
        rel_ids = [r.id for r in relations]

        assert "born_in" in rel_ids
        assert "field" in rel_ids

    def test_get_triples_by_subject(self, sample_kg):
        """Test querying triples by subject."""
        triples = sample_kg.get_triples(subject="Einstein")
        assert len(triples) == 2
        assert all(t.subject_id == "Einstein" for t in triples)

    def test_get_triples_by_predicate(self, sample_kg):
        """Test querying triples by predicate."""
        triples = sample_kg.get_triples(predicate="field")
        assert len(triples) == 3  # Einstein Physics, Curie Physics, Curie Chemistry

    def test_get_triples_by_object(self, sample_kg):
        """Test querying triples by object."""
        triples = sample_kg.get_triples(obj="Physics")
        assert len(triples) == 2

    def test_get_subgraph(self, sample_kg):
        """Test extracting a subgraph."""
        subgraph = sample_kg.get_subgraph(
            entity_ids=["Einstein"],
            max_hops=1,
            max_triples=10
        )

        assert subgraph.size >= 2
        assert "Einstein" in subgraph.entities
        assert "Ulm" in subgraph.entities or "Physics" in subgraph.entities

    def test_find_paths(self, sample_kg):
        """Test finding paths between entities."""
        paths = sample_kg.find_paths(
            source="Einstein",
            target="Germany",
            max_hops=2
        )

        # Einstein -> born_in -> Ulm -> located_in -> Germany
        assert len(paths) >= 1
        if paths:
            assert len(paths[0]) == 2

    def test_remove_triple(self, sample_kg):
        """Test removing a triple."""
        initial_count = sample_kg.num_triples
        triple = Triple("Einstein", "born_in", "Ulm")

        result = sample_kg.remove_triple(triple)

        assert result is True
        assert sample_kg.num_triples == initial_count - 1

    def test_remove_entity(self, sample_kg):
        """Test removing an entity and its triples."""
        initial_triples = sample_kg.num_triples

        result = sample_kg.remove_entity("Einstein")

        assert result is True
        assert sample_kg.num_triples < initial_triples
        assert sample_kg.get_entity("Einstein") is None

    def test_clear(self, sample_kg):
        """Test clearing the KG."""
        sample_kg.clear()

        assert sample_kg.num_entities == 0
        assert sample_kg.num_triples == 0

    def test_from_triples_classmethod(self):
        """Test creating KG from list of triples."""
        triples = [
            Triple("A", "r1", "B"),
            Triple("B", "r2", "C"),
        ]
        kg = InMemoryKG.from_triples(triples, name="FromTriples")

        assert kg.num_triples == 2
        assert kg.name == "FromTriples"

    def test_from_tuples(self):
        """Test creating KG from tuples."""
        tuples = [
            ("A", "r1", "B"),
            ("B", "r2", "C"),
        ]
        kg = InMemoryKG.from_triples(tuples)

        assert kg.num_triples == 2

    def test_add_entity_with_metadata(self, empty_kg):
        """Test adding an entity with full metadata."""
        entity = Entity(
            id="Q42",
            name="Douglas Adams",
            aliases=["DNA"],
            description="Author",
            entity_type=EntityType.PERSON,
            properties={"birth_year": 1952}
        )
        empty_kg.add_entity(entity)

        retrieved = empty_kg.get_entity("Q42")
        assert retrieved is not None
        assert retrieved.name == "Douglas Adams"
        assert "DNA" in retrieved.aliases

    def test_summary(self, sample_kg):
        """Test summary generation."""
        summary = sample_kg.summary()

        assert "SampleKG" in summary
        assert "Entities" in summary
        assert "Triples" in summary

    def test_repr(self, sample_kg):
        """Test string representation."""
        repr_str = repr(sample_kg)

        assert "InMemoryKG" in repr_str
        assert "entities=" in repr_str
        assert "triples=" in repr_str
