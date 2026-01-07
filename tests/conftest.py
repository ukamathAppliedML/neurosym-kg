"""
Pytest configuration and shared fixtures.
"""

import pytest
from neurosym_kg import (
    InMemoryKG,
    Triple,
    MockLLMBackend,
    Entity,
    EntityType,
)


@pytest.fixture
def simple_kg():
    """Create a simple test KG."""
    kg = InMemoryKG(name="TestKG")
    kg.add_triples([
        Triple("Einstein", "born_in", "Ulm"),
        Triple("Einstein", "field", "Physics"),
        Triple("Einstein", "nationality", "German"),
        Triple("Ulm", "located_in", "Germany"),
        Triple("Curie", "born_in", "Warsaw"),
        Triple("Curie", "field", "Physics"),
        Triple("Warsaw", "located_in", "Poland"),
    ])
    return kg


@pytest.fixture
def movie_kg():
    """Create a movie-themed test KG."""
    kg = InMemoryKG(name="MovieKG")
    kg.add_triples([
        Triple("Inception", "director", "Christopher_Nolan"),
        Triple("Inception", "year", "2010"),
        Triple("Inception", "starring", "Leonardo_DiCaprio"),
        Triple("The_Dark_Knight", "director", "Christopher_Nolan"),
        Triple("The_Dark_Knight", "year", "2008"),
        Triple("Christopher_Nolan", "born_in", "London"),
        Triple("Christopher_Nolan", "nationality", "British"),
        Triple("Leonardo_DiCaprio", "born_in", "Los_Angeles"),
    ])
    return kg


@pytest.fixture
def mock_llm():
    """Create a mock LLM backend."""
    llm = MockLLMBackend(default_response="I don't know.")
    
    # Common patterns
    llm.add_response(r".*Extract.*entities.*", "Einstein")
    llm.add_response(r".*relevant relations.*", "born_in")
    llm.add_response(r".*enough information.*", "YES")
    llm.add_response(r".*answer.*", "The answer is Ulm.")
    
    return llm


@pytest.fixture
def detailed_mock_llm():
    """Create a mock LLM with more detailed responses."""
    llm = MockLLMBackend()
    
    # Entity extraction
    llm.add_response(r".*Extract.*Einstein.*", "Einstein")
    llm.add_response(r".*Extract.*Curie.*", "Curie")
    llm.add_response(r".*Extract.*Inception.*", "Inception")
    llm.add_response(r".*Extract.*Nolan.*", "Christopher_Nolan")
    
    # Relation selection
    llm.add_response(r".*relations.*born.*", "born_in")
    llm.add_response(r".*relations.*direct.*", "director")
    llm.add_response(r".*relations.*field.*", "field")
    
    # Reasoning checks
    llm.add_response(r".*enough information.*", "YES - Sufficient information found.")
    
    # Answer generation
    llm.add_response(r".*answer.*born.*Einstein.*", "Einstein was born in Ulm.")
    llm.add_response(r".*answer.*director.*Inception.*", "Christopher Nolan directed Inception.")
    llm.add_response(r".*answer.*field.*Einstein.*", "Physics")
    
    return llm


@pytest.fixture
def sample_entity():
    """Create a sample entity."""
    return Entity(
        id="Q42",
        name="Douglas Adams",
        aliases=["DNA", "Douglas Noel Adams"],
        description="English author and screenwriter",
        entity_type=EntityType.PERSON,
        properties={"birth_year": 1952, "death_year": 2001},
    )


@pytest.fixture
def sample_triples():
    """Create a list of sample triples."""
    return [
        Triple("A", "r1", "B"),
        Triple("B", "r2", "C"),
        Triple("C", "r3", "D"),
        Triple("A", "r4", "E"),
    ]


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Fast unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may need API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests"
    )
