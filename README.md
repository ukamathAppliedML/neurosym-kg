# NeuroSym-KG: A Unified Neuro-Symbolic Reasoning Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, extensible framework for neuro-symbolic AI that unifies Knowledge Graphs (KGs) with Large Language Models (LLMs). This framework implements state-of-the-art reasoning paradigms including Think-on-Graph (ToG), Reasoning on Graphs (RoG), GraphRAG, and SubgraphRAG, with pluggable backends for both knowledge sources and language models.

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Background & Motivation](#-background--motivation)
- [Architecture](#️-architecture)
- [Supported Backends](#-supported-backends)
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Reasoning Paradigms](#-reasoning-paradigms)
- [Knowledge Graph Setup](#️-knowledge-graph-setup)
- [LLM Backend Setup](#-llm-backend-setup)
- [Evaluation & Benchmarking](#-evaluation--benchmarking)
  - [Supported Datasets](#supported-benchmark-datasets)
  - [Running Benchmarks](#running-benchmarks)
  - [Mintaka Dataset](#mintaka-dataset-format)
  - [Evaluation Metrics](#evaluation-metrics)
- [Testing](#-testing)
  - [Smoke Test](#smoke-test-no-dependencies)
  - [Unit Tests](#unit-tests)
- [Quick Reference](#-quick-reference)
- [Troubleshooting](#-troubleshooting)
- [Configuration](#-configuration)
- [Notebooks](#-notebooks)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [Citation](#-citation)

## 🎯 Key Features

- **Unified Abstraction Layer**: Common interfaces for KGs, LLMs, and reasoning strategies
- **Multiple KG Backends**: Wikidata, Neo4j, in-memory graphs, and custom connectors
- **Pluggable Reasoners**: ToG, RoG, GraphRAG, SubgraphRAG, and extensible base classes
- **LLM Agnostic**: OpenAI, Anthropic Claude, Ollama (local), and extensible backends
- **Local-First Option**: Run entirely on your machine with Ollama + InMemoryKG/Neo4j
- **Symbolic Integration**: Rule engines and constraint verification
- **Comprehensive Benchmarks**: Mintaka, SimpleQuestions, LC-QuAD 2.0, Wikidata
- **Evaluation Suite**: Accuracy, F1, Hits@K, faithfulness, latency metrics
- **Production Ready**: Caching, rate limiting, comprehensive logging

## 📚 Background & Motivation

This framework is inspired by two foundational surveys:

1. **"Unifying Large Language Models and Knowledge Graphs: A Roadmap"** (TKDE 2024)
   - Pan et al. - Comprehensive taxonomy of LLM-KG integration approaches
   
2. **"Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey"** (IEEE TNNLS 2024)
   - DeLong et al. - Classification of neurosymbolic reasoning methods

### The Integration Paradigms

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Neuro-Symbolic Integration Spectrum                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM ⊕ KG (Loose)          LLM ⊗ KG (Tight)         Differentiable     │
│  ─────────────────         ────────────────         ──────────────      │
│  KG augments prompts       LLM as agent on KG      End-to-end training  │
│  • RAG + KG retrieval      • Think-on-Graph        • Logic Tensor Nets  │
│  • GraphRAG                • Reasoning on Graphs   • DeepProbLog        │
│  • KG-enhanced prompts     • Interactive traversal • Neural Theorem     │
│                                                      Provers            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

```
neurosym_kg/
├── core/                    # Abstract interfaces & base classes
│   ├── interfaces.py        # Protocol definitions
│   ├── types.py            # Core data types (Entity, Triple, etc.)
│   ├── config.py           # Configuration management
│   └── exceptions.py       # Custom exceptions
│
├── knowledge_graphs/        # KG backend implementations
│   ├── base.py             # Abstract KG interface
│   ├── in_memory.py        # In-memory graph for testing/demos
│   ├── wikidata.py         # Wikidata SPARQL connector
│   └── neo4j_kg.py         # Neo4j graph database
│
├── reasoners/               # Reasoning paradigm implementations
│   ├── base.py             # Abstract reasoner interface
│   ├── think_on_graph.py   # ToG: Beam search with LLM guidance
│   ├── reasoning_on_graphs.py  # RoG: Faithful planning paths
│   ├── graph_rag.py        # GraphRAG: Community-based retrieval
│   └── subgraph_rag.py     # SubgraphRAG: Flexible subgraph retrieval
│
├── llm_backends/            # LLM provider implementations
│   ├── base.py             # Abstract LLM interface
│   ├── openai_backend.py   # OpenAI API (GPT-4, GPT-4o)
│   ├── anthropic_backend.py # Anthropic API (Claude)
│   ├── ollama_backend.py   # Ollama (local models)
│   └── mock_backend.py     # Mock LLM for testing
│
├── symbolic/                # Symbolic reasoning modules
│   ├── rules.py            # Rule representation & matching
│   └── constraint_checker.py # Output verification against KG
│
├── evaluation/              # Benchmarking & metrics
│   ├── benchmarks.py       # WebQSP, CWQ, HotpotQA loaders
│   ├── metrics.py          #  Accuracy, F1, Hits@K, faithfulness
│   └── runner.py           # Evaluation orchestration
│
└── utils/                   # Shared utilities
    └── caching.py          # Response & embedding caching
```

## 🔌 Supported Backends

### Knowledge Graph Backends

| Backend | Description | Setup Required | Best For |
|---------|-------------|----------------|----------|
| **InMemoryKG** | Fast in-memory graph | None | Testing, demos, small graphs |
| **WikidataKG** | Live Wikidata via SPARQL | None | Real-world facts, 100M+ entities |
| **Neo4jKG** | Neo4j graph database | Neo4j server | Production, custom data, enterprise |

### LLM Backends

| Backend | Models | Setup Required | Best For |
|---------|--------|----------------|----------|
| **OllamaBackend** | Llama, Qwen, Mistral, etc. | `ollama serve` | Local inference, privacy, no API costs |
| **OpenAIBackend** | GPT-4, GPT-4o, GPT-4o-mini | API key | Highest quality, production |
| **AnthropicBackend** | Claude Sonnet, Opus, Haiku | API key | Long context, complex reasoning |
| **MockLLMBackend** | N/A | None | Testing, development |

### Reasoning Paradigms

| Reasoner | Approach | Strengths |
|----------|----------|-----------|
| **ThinkOnGraph** | Beam search with LLM-guided exploration | Most robust, handles ambiguity well |
| **ReasoningOnGraphs** | Plan-based faithful reasoning | Explainable paths, faithful to KG |
| **GraphRAG** | Community-based retrieval | Good for summarization, global queries |
| **SubgraphRAG** | Flexible subgraph retrieval | Balance of context and precision |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/neurosym-kg.git
cd neurosym-kg

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install with all dependencies
pip install -e ".[all]"

# Or install with specific backends only
pip install -e ".[openai]"           # OpenAI support
pip install -e ".[anthropic]"        # Anthropic support  
pip install -e ".[neo4j]"            # Neo4j support
pip install -e ".[dev]"              # Development tools
```

### Option A: Local Setup (Ollama - No API Keys!)

The easiest way to get started - runs entirely on your machine:

```bash
# 1. Install Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
# Or download from: https://ollama.ai

# 2. Start Ollama server
ollama serve

# 3. Pull a model (in another terminal)
ollama pull qwen2.5-coder:7b   # Recommended for reasoning
# or: ollama pull llama3.2     # Alternative

# 4. Run the demo
python examples/ollama_demo.py
```

### Option B: Cloud LLMs (OpenAI/Anthropic)

```bash
# Set API key
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."

# Run
python examples/basic_qa.py
```

### Basic Usage

```python
from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.llm_backends import OllamaBackend
from neurosym_kg.reasoners import ThinkOnGraphReasoner

# Create a knowledge graph
kg = InMemoryKG(name="Scientists")
kg.add_triples([
    Triple("Einstein", "born_in", "Ulm"),
    Triple("Einstein", "field", "Physics"),
    Triple("Ulm", "located_in", "Germany"),
])

# Connect to local LLM
llm = OllamaBackend(model="qwen2.5-coder:7b")

# Create reasoner
reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm, max_depth=3)

# Ask questions!
result = reasoner.reason("Where was Einstein born?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
```

### Swapping Components

```python
# Use Neo4j instead of InMemoryKG
from neurosym_kg.knowledge_graphs import Neo4jKG
kg = Neo4jKG(uri="bolt://localhost:7687", username="neo4j", password="password")

# Use Claude instead of Ollama
from neurosym_kg.llm_backends import AnthropicBackend
llm = AnthropicBackend(model="claude-sonnet-4-20250514")

# Use different reasoner
from neurosym_kg.reasoners import ReasoningOnGraphs
reasoner = ReasoningOnGraphs(kg=kg, llm=llm)
```

## 📖 Examples

The `examples/` directory contains complete working demos:

### Core Demos

| Example | Description | Requirements |
|---------|-------------|--------------|
| `ollama_demo.py` | Basic KG + Ollama reasoning | Ollama |
| `compare_reasoners_ollama.py` | Compare all 4 reasoners | Ollama |
| `basic_qa.py` | Simple question answering | Any LLM |

### Knowledge Graph Demos

| Example | Description | Requirements |
|---------|-------------|--------------|
| `neo4j_example.py` | Neo4j CRUD operations | Neo4j |
| `neo4j_ollama_demo.py` | Neo4j + Ollama reasoning | Neo4j + Ollama |

### Benchmark Demos

| Example | Description | Requirements |
|---------|-------------|--------------|
| `benchmark_real.py` | Mintaka/SimpleQuestions/LC-QuAD benchmarks | Ollama + Internet |
| `benchmark_wikidata.py` | Live Wikidata QA benchmark | Ollama |
| `benchmark_ollama.py` | Quick local benchmark | Ollama |

### LLM Backend Demos

| Example | Description | Requirements |
|---------|-------------|--------------|
| `anthropic_example.py` | Anthropic Claude usage | Anthropic API key |

### Running Examples

```bash
# Activate environment
source venv/bin/activate

# Run with Ollama (local, free)
ollama serve  # In another terminal
python examples/ollama_demo.py
python examples/compare_reasoners_ollama.py --model qwen2.5-coder:7b

# Run benchmarks
python examples/benchmark_real.py --dataset mintaka --num-examples 50
python examples/benchmark_wikidata.py

# Run with Neo4j + Ollama
python examples/neo4j_ollama_demo.py

# Run with OpenAI
export OPENAI_API_KEY="sk-..."
python examples/basic_qa.py
```

## 🧠 Reasoning Paradigms

### Think-on-Graph (ToG)
*From: Sun et al., ICLR 2024*

LLM acts as an agent, navigating the KG through beam search:

```python
from neurosym_kg.reasoners import ThinkOnGraphReasoner

reasoner = ThinkOnGraphReasoner(
    kg=kg,
    llm=llm,
    max_depth=3,      # Maximum hops from seed entity
    beam_width=5,     # Candidates to explore at each step
    verbose=True,     # Show exploration process
)

result = reasoner.reason("Who directed Inception?")
# LLM explores: Inception → director → Christopher_Nolan
```

### Reasoning on Graphs (RoG)
*From: Luo et al., ICLR 2024*

Plan-based faithful reasoning with explicit path generation:

```python
from neurosym_kg.reasoners import ReasoningOnGraphs

reasoner = ReasoningOnGraphs(
    kg=kg,
    llm=llm,
    plan_beam_width=3,
    max_plan_length=4,
)

result = reasoner.reason("What country was Einstein born in?")
# Plan: Einstein → born_in → ? → located_in → Country
```

### GraphRAG
*From: Microsoft Research, 2024*

Community-based retrieval for global queries:

```python
from neurosym_kg.reasoners import GraphRAGReasoner

reasoner = GraphRAGReasoner(
    kg=kg,
    llm=llm,
    community_algorithm="leiden",
    summary_max_tokens=500,
)
```

### SubgraphRAG
*From: Li et al., 2024*

Flexible subgraph retrieval with adjustable granularity:

```python
from neurosym_kg.reasoners import SubgraphRAGReasoner

reasoner = SubgraphRAGReasoner(
    kg=kg,
    llm=llm,
    subgraph_size=50,
    retrieval_method="hybrid",
)
```

## 🗄️ Knowledge Graph Setup

### InMemoryKG (No Setup Required)

```python
from neurosym_kg import InMemoryKG, Triple

kg = InMemoryKG(name="My KG")
kg.add_triples([
    Triple("subject", "predicate", "object"),
])
```

### WikidataKG (No Setup Required)

```python
from neurosym_kg.knowledge_graphs import WikidataKG

kg = WikidataKG()
# Queries live Wikidata SPARQL endpoint
neighbors = kg.get_neighbors("Q937")  # Albert Einstein
```

### Neo4jKG (Requires Neo4j Server)

**Option 1: Neo4j Desktop (Recommended)**
1. Download from https://neo4j.com/download/
2. Create a new project and database
3. Start the database
4. Note the password you set

**Option 2: Docker**
```bash
docker run -d \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  --name neo4j \
  neo4j:latest
```

**Connect from Python:**
```python
from neurosym_kg.knowledge_graphs import Neo4jKG

kg = Neo4jKG(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",  # Your password
)

# Add data
kg.add_triples([Triple("Einstein", "born_in", "Ulm")])

# Query
neighbors = kg.get_neighbors("Einstein")
path = kg.get_shortest_path("Einstein", "Germany")

# Don't forget to close
kg.close()
```

## 🤖 LLM Backend Setup

### OllamaBackend (Local - Recommended for Development)

```bash
# Install and start Ollama
ollama serve

# Pull models
ollama pull qwen2.5-coder:7b   # Best for reasoning tasks
ollama pull llama3.2            # Good general model
ollama pull mistral             # Fast alternative
```

```python
from neurosym_kg.llm_backends import OllamaBackend

llm = OllamaBackend(
    model="qwen2.5-coder:7b",
    temperature=0.1,  # Lower = more deterministic
)

# List available models
print(llm.list_models())
```

### OpenAIBackend

```python
from neurosym_kg.llm_backends import OpenAIBackend

llm = OpenAIBackend(
    model="gpt-4o-mini",  # or "gpt-4o", "gpt-4"
    # api_key="sk-..."  # Or set OPENAI_API_KEY env var
)
```

### AnthropicBackend

```python
from neurosym_kg.llm_backends import AnthropicBackend

llm = AnthropicBackend(
    model="claude-sonnet-4-20250514",  # or "claude-opus-4-20250514"
    # api_key="sk-ant-..."  # Or set ANTHROPIC_API_KEY env var
)
```

## 📊 Evaluation & Benchmarking

NeuroSym-KG includes a comprehensive evaluation system for testing reasoners against standard QA benchmarks.

### Supported Benchmark Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| **Mintaka** | Amazon Science | ~20K | Multilingual complex QA over Wikidata |
| **SimpleQuestions** | Wikidata version | 100K+ | Single-hop factual questions |
| **LC-QuAD 2.0** | SDA Research | 30K | Simple and complex Wikidata questions |
| **Fallback** | Built-in | 100+ | Curated questions when downloads fail |

### Running Benchmarks

#### Quick Start: Mintaka Benchmark

```bash
# Start Ollama
ollama serve

# Run Mintaka benchmark (default)
python examples/benchmark_real.py --num-examples 50

# With verbose output
python examples/benchmark_real.py --num-examples 30 --verbose

# Compare all reasoners
python examples/benchmark_real.py --compare-all --num-examples 30
```

#### Benchmark Options

```bash
# Choose dataset
python examples/benchmark_real.py --dataset mintaka        # Default
python examples/benchmark_real.py --dataset simple-questions
python examples/benchmark_real.py --dataset lcquad2
python examples/benchmark_real.py --dataset fallback       # No download needed

# Choose reasoner
python examples/benchmark_real.py --reasoner tog           # Think-on-Graph
python examples/benchmark_real.py --reasoner graphrag      # GraphRAG
python examples/benchmark_real.py --reasoner subgraphrag   # SubgraphRAG

# Choose model
python examples/benchmark_real.py --model qwen2.5-coder:7b
python examples/benchmark_real.py --model llama3.2

# Save results
python examples/benchmark_real.py --output results.json
```

### How Benchmarking Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Benchmark Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Download Dataset                                                     │
│     └── mintaka_test.json from GitHub → data/benchmarks/                │
│                                                                          │
│  2. Parse Questions                                                      │
│     └── Extract: question, expected_answer, entities                    │
│                                                                          │
│  3. Connect to KG                                                        │
│     └── WikidataKG (live SPARQL) or InMemoryKG                          │
│                                                                          │
│  4. Run Reasoner                                                         │
│     └── ToG/RoG/GraphRAG/SubgraphRAG + LLM                              │
│                                                                          │
│  5. Evaluate                                                             │
│     └── Compare predicted vs expected (exact match + F1)                │
│                                                                          │
│  6. Report                                                               │
│     └── Accuracy, F1, Latency, Error analysis                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### Mintaka Dataset Format

The Mintaka dataset (from Amazon Science) contains complex QA pairs:

```json
{
  "question": "Who directed Inception?",
  "answer": {
    "mention": "Christopher Nolan"
  },
  "questionEntity": [
    {"name": "Inception", "entityType": "Q11424"}
  ],
  "complexityType": "simple"
}
```

### Example Benchmark Output

```
==================================================
NeuroSym-KG Real Benchmark
==================================================
Dataset:    mintaka
Model:      qwen2.5-coder:7b
Examples:   50

1. Testing Ollama...
   ✅ Connected

2. Loading dataset: mintaka
   Downloading https://raw.githubusercontent.com/.../mintaka_test.json
   ✅ Loaded 50 examples

3. Connecting to Wikidata...
   ✅ Connected

==================================================
Evaluating: ToG
Examples: 50
==================================================
  [1/50] Who directed Inception?... ✅
  [2/50] What is the capital of France?... ✅
  [3/50] When was Albert Einstein born?... ✅
  ...

==================================================
RESULTS: ToG
==================================================
  Examples:     50
  Correct:      35
  Accuracy:     70.0%
  Avg F1:       0.750
  Avg Latency:  1200ms

  Sample Errors:
    Q: What year did the Roman Empire fall?...
    Got: 476 AD...
    Expected: ['476']
```

### Comparing Reasoners

```bash
python examples/benchmark_real.py --compare-all --num-examples 50
```

Output:
```
==================================================
COMPARISON
==================================================
Reasoner             Accuracy         F1      Latency
------------------------------------------------------------
ToG                     70.0%      0.750       1200ms
GraphRAG                65.0%      0.720       1500ms
SubgraphRAG             68.0%      0.735       1100ms
```

### Evaluation Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Exact Match** | Answer exactly matches expected | 0-100% |
| **F1 Score** | Token overlap between predicted/expected | 0.0-1.0 |
| **Hits@K** | Correct answer in top K predictions | 0-100% |
| **Faithfulness** | Answer grounded in retrieved KG facts | 0-100% |
| **Path Validity** | Reasoning paths are valid KG paths | 0-100% |
| **Latency** | Response time (p50, p95, avg) | ms |

### Programmatic Evaluation

```python
from neurosym_kg.evaluation import Metrics, EvaluationRunner

# Simple metrics
accuracy = Metrics.accuracy(predictions, ground_truth)
f1 = Metrics.f1_score(predicted_text, expected_text)
exact = Metrics.exact_match(pred, expected)

# Full evaluation runner
from neurosym_kg.evaluation import EvaluationRunner

runner = EvaluationRunner(reasoner)
results = runner.evaluate(
    questions=questions,
    answers=expected_answers,
    verbose=True,
)

print(f"Accuracy: {results['accuracy']:.1%}")
print(f"Avg F1: {results['avg_f1']:.3f}")
print(f"Avg Latency: {results['avg_latency_ms']:.0f}ms")
```

### Wikidata Benchmark (Live Queries)

Test against live Wikidata for real-world performance:

```bash
python examples/benchmark_wikidata.py --model qwen2.5-coder:7b
```

This runs 20 curated questions against the live Wikidata SPARQL endpoint.


## 🔧 Configuration

### Environment Variables

```bash
# LLM API Keys (only needed for cloud LLMs)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Neo4j Connection (only if using Neo4j)
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# Caching
export NEUROSYM_CACHE_DIR="~/.cache/neurosym_kg"
```

### Programmatic Configuration

```python
from neurosym_kg import Config

config = Config(
    cache_enabled=True,
    cache_dir="~/.cache/neurosym_kg",
    log_level="INFO",
    timeout_seconds=30,
)

reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm, config=config)
```

## ⚡ Quick Reference

### Command Cheat Sheet

```bash
# === SETUP ===
pip install -e ".[all]"              # Install all dependencies
ollama serve                          # Start Ollama server
ollama pull qwen2.5-coder:7b         # Download recommended model

# === VERIFY INSTALLATION ===
python smoke_test.py                  # Run smoke test (no external deps)
pytest tests/unit/ -v                 # Run unit tests

# === RUN DEMOS ===
python examples/ollama_demo.py                    # Basic demo
python examples/compare_reasoners_ollama.py       # Compare reasoners
python examples/benchmark_wikidata.py             # Wikidata benchmark

# === RUN BENCHMARKS ===
python examples/benchmark_real.py                           # Mintaka (default)
python examples/benchmark_real.py --dataset simple-questions
python examples/benchmark_real.py --compare-all --num-examples 30
python examples/benchmark_real.py --output results.json

# === NEO4J ===
python examples/neo4j_example.py                  # Neo4j CRUD demo
python examples/neo4j_ollama_demo.py              # Neo4j + Ollama reasoning
```

### Python Quick Start

```python
# Minimal example - InMemoryKG + Ollama + ToG
from neurosym_kg import InMemoryKG, Triple
from neurosym_kg.llm_backends import OllamaBackend
from neurosym_kg.reasoners import ThinkOnGraphReasoner

kg = InMemoryKG()
kg.add_triples([Triple("Einstein", "born_in", "Ulm")])
llm = OllamaBackend(model="qwen2.5-coder:7b")
reasoner = ThinkOnGraphReasoner(kg=kg, llm=llm)
result = reasoner.reason("Where was Einstein born?")
print(result.answer)  # "Ulm"
```

### Data Directory Structure

After running benchmarks, this structure is created:

```
data/
└── benchmarks/
    ├── mintaka_test.json           # Downloaded Mintaka dataset
    ├── simple_questions_wikidata_test.txt  # SimpleQuestions
    └── lcquad2_test.json           # LC-QuAD 2.0
```


### Performance Tips

| Tip | Impact |
|-----|--------|
| Use `qwen2.5-coder:7b` | Best accuracy/speed balance |
| Lower `temperature=0.1` | More deterministic answers |
| Reduce `beam_width=3` | Faster but less thorough |
| Reduce `max_depth=2` | Faster for simple questions |
| Enable caching | Avoid repeated LLM calls |
| Use InMemoryKG for <10K triples | Fastest for small graphs |
| Use Neo4j for >10K triples | Scales better |

## 📓 Notebooks

Interactive Jupyter notebooks in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `01_quickstart.ipynb` | Installation, basic KG creation, first reasoning query |
| `02_comparing_reasoners.ipynb` | Side-by-side comparison of ToG, RoG, GraphRAG, SubgraphRAG |
| `04_evaluation.ipynb` | Running WebQSP/CWQ benchmarks |
| `05_symbolic_reasoning.ipynb` | Rule engines, constraint checking |

```bash
# Run notebooks locally
pip install jupyter
jupyter notebook notebooks/
```


## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Development setup
git clone https://github.com/your-org/neurosym-kg.git
cd neurosym-kg
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,all]"
pre-commit install

# Run quality checks
black neurosym_kg tests
ruff check neurosym_kg tests
mypy neurosym_kg
pytest tests/unit/ -v
```

## 📄 Citation

```bibtex
@software{neurosym_kg,
  title = {NeuroSym-KG: A Unified Neuro-Symbolic Reasoning Framework},
  year = {2024},
  url = {https://github.com/ukamathAppliedML/neurosym-kg}
}
```

### Key References

```bibtex
@article{pan2024unifying,
  title={Unifying Large Language Models and Knowledge Graphs: A Roadmap},
  author={Pan, Shirui and others},
  journal={IEEE TKDE},
  year={2024}
}

@inproceedings{sun2024think,
  title={Think-on-Graph: Deep and Responsible Reasoning of LLM on Knowledge Graph},
  author={Sun, Jiashuo and others},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{luo2024reasoning,
  title={Reasoning on Graphs: Faithful and Interpretable LLM Reasoning},
  author={Luo, Linhao and others},
  booktitle={ICLR},
  year={2024}
}
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.
