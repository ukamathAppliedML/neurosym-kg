"""
Unit tests for core types.
"""

import pytest
from neurosym_kg.core.types import (
    Entity,
    EntityType,
    Relation,
    RelationType,
    Triple,
    ReasoningPath,
    Subgraph,
    ReasoningResult,
    ReasoningResultStatus,
    Message,
)


class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        entity = Entity(id="Q42", name="Douglas Adams")
        assert entity.id == "Q42"
        assert entity.name == "Douglas Adams"
        assert entity.entity_type == EntityType.UNKNOWN
        assert entity.aliases == []

    def test_entity_with_all_fields(self):
        entity = Entity(
            id="Q42",
            name="Douglas Adams",
            aliases=["DNA", "Douglas Noel Adams"],
            description="English author",
            entity_type=EntityType.PERSON,
            properties={"birth_year": 1952},
        )
        assert entity.aliases == ["DNA", "Douglas Noel Adams"]
        assert entity.description == "English author"
        assert entity.entity_type == EntityType.PERSON
        assert entity.properties["birth_year"] == 1952

    def test_entity_equality(self):
        e1 = Entity(id="Q42", name="Douglas Adams")
        e2 = Entity(id="Q42", name="D. Adams")  # Same ID, different name
        e3 = Entity(id="Q43", name="Douglas Adams")  # Different ID

        assert e1 == e2  # Same ID
        assert e1 != e3  # Different ID

    def test_entity_hash(self):
        e1 = Entity(id="Q42", name="Douglas Adams")
        e2 = Entity(id="Q42", name="D. Adams")

        # Should be usable in sets/dicts
        entity_set = {e1, e2}
        assert len(entity_set) == 1


class TestRelation:
    """Tests for Relation class."""

    def test_relation_creation(self):
        rel = Relation(id="P31", name="instance of")
        assert rel.id == "P31"
        assert rel.name == "instance of"
        assert rel.relation_type == RelationType.OTHER

    def test_relation_types(self):
        rel = Relation(
            id="P17",
            name="country",
            relation_type=RelationType.SPATIAL,
        )
        assert rel.relation_type == RelationType.SPATIAL


class TestTriple:
    """Tests for Triple class."""

    def test_triple_with_strings(self):
        triple = Triple(subject="Einstein", predicate="born_in", object="Ulm")
        assert triple.subject_id == "Einstein"
        assert triple.predicate_id == "born_in"
        assert triple.object_id == "Ulm"

    def test_triple_with_entities(self):
        subj = Entity(id="Q937", name="Einstein")
        pred = Relation(id="P19", name="place of birth")
        obj = Entity(id="Q3012", name="Ulm")

        triple = Triple(subject=subj, predicate=pred, object=obj)

        assert triple.subject_id == "Q937"
        assert triple.predicate_id == "P19"
        assert triple.object_id == "Q3012"

    def test_triple_to_tuple(self):
        triple = Triple(subject="A", predicate="rel", object="B")
        assert triple.to_tuple() == ("A", "rel", "B")

    def test_triple_to_text(self):
        triple = Triple(subject="Einstein", predicate="born_in", object="Ulm")
        text = triple.to_text()
        assert "Einstein" in text
        assert "born_in" in text
        assert "Ulm" in text


class TestReasoningPath:
    """Tests for ReasoningPath class."""

    def test_empty_path(self):
        path = ReasoningPath()
        assert path.length == 0
        assert path.relations == []

    def test_path_with_triples(self):
        t1 = Triple(subject="A", predicate="r1", object="B")
        t2 = Triple(subject="B", predicate="r2", object="C")

        path = ReasoningPath(triples=[t1, t2], score=0.9)

        assert path.length == 2
        assert path.relations == ["r1", "r2"]
        assert path.score == 0.9

    def test_path_to_text(self):
        t1 = Triple(subject="A", predicate="r1", object="B")
        path = ReasoningPath(triples=[t1])

        text = path.to_text()
        assert "A" in text
        assert "B" in text


class TestSubgraph:
    """Tests for Subgraph class."""

    def test_subgraph_auto_entities(self):
        t1 = Triple(subject="A", predicate="r1", object="B")
        t2 = Triple(subject="B", predicate="r2", object="C")

        subgraph = Subgraph(triples=[t1, t2])

        # Entities should be auto-populated
        assert "A" in subgraph.entities
        assert "B" in subgraph.entities
        assert "C" in subgraph.entities
        assert subgraph.size == 2


class TestReasoningResult:
    """Tests for ReasoningResult class."""

    def test_successful_result(self):
        result = ReasoningResult(
            answer="Paris",
            status=ReasoningResultStatus.SUCCESS,
            confidence=0.95,
        )
        assert result.is_successful
        assert result.primary_answer == "Paris"

    def test_list_answer(self):
        result = ReasoningResult(
            answer=["Paris", "Lyon"],
            status=ReasoningResultStatus.SUCCESS,
        )
        assert result.primary_answer == "Paris"

    def test_failed_result(self):
        result = ReasoningResult(
            answer="",
            status=ReasoningResultStatus.NO_ANSWER,
        )
        assert not result.is_successful


class TestMessage:
    """Tests for Message class."""

    def test_message_to_dict(self):
        msg = Message(role="user", content="Hello!")
        d = msg.to_dict()
        assert d == {"role": "user", "content": "Hello!"}
