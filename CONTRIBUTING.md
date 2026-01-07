# Contributing to NeuroSym-KG

Thank you for your interest in contributing to NeuroSym-KG! This document provides guidelines and instructions for contributing.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR-USERNAME/neurosym-kg.git
cd neurosym-kg
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Workflow

### Code Style

We use:
- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

Run all checks:
```bash
# Format code
black neurosym_kg tests
isort neurosym_kg tests

# Lint
ruff check neurosym_kg tests

# Type check
mypy neurosym_kg
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=neurosym_kg --cov-report=html

# Run specific test file
pytest tests/unit/test_in_memory_kg.py -v

# Run tests matching a pattern
pytest -k "test_entity" -v
```

### Documentation

- Add docstrings to all public functions and classes
- Follow Google-style docstring format
- Update README.md if adding new features
- Add examples for new functionality

## Contribution Types

### Bug Reports

Open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS, etc.)

### Feature Requests

Open an issue with:
- Description of the feature
- Use case / motivation
- Proposed API (if applicable)

### Pull Requests

1. Ensure all tests pass
2. Add tests for new functionality
3. Update documentation
4. Follow the code style guidelines
5. Write clear commit messages

## Adding New Components

### New Reasoner

1. Create `neurosym_kg/reasoners/your_reasoner.py`
2. Inherit from `BaseReasoner`
3. Implement `reason()` and `areason()` methods
4. Add to `neurosym_kg/reasoners/__init__.py`
5. Add tests in `tests/unit/test_your_reasoner.py`
6. Add example in `examples/`

Example structure:
```python
from neurosym_kg.reasoners.base import BaseReasoner

class YourReasoner(BaseReasoner):
    """Your reasoner description."""
    
    def __init__(self, kg, llm, **kwargs):
        super().__init__(kg, llm, name="YourReasoner")
        # Initialize your reasoner
    
    def reason(self, question: str, context: str = None, **kwargs):
        # Implement reasoning logic
        pass
```

### New KG Backend

1. Create `neurosym_kg/knowledge_graphs/your_kg.py`
2. Inherit from `BaseKnowledgeGraph` or `BaseMutableKnowledgeGraph`
3. Implement all required methods
4. Add to `neurosym_kg/knowledge_graphs/__init__.py`
5. Add tests

### New LLM Backend

1. Create `neurosym_kg/llm_backends/your_backend.py`
2. Inherit from `BaseLLMBackend`
3. Implement `generate()` and `agenerate()` methods
4. Add to `neurosym_kg/llm_backends/__init__.py`
5. Add tests

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat: add Neo4j knowledge graph backend
fix: handle empty response in ToG reasoner
docs: add Colab notebook for evaluation
test: add integration tests for Wikidata
```

## Review Process

1. Open a pull request
2. Wait for CI checks to pass
3. Address reviewer feedback
4. Once approved, maintainers will merge

## Questions?

- Open a Discussion on GitHub
- Tag maintainers in issues if urgent

Thank you for contributing! 🎉
