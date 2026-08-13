# Production RAG with Hybrid Search

This project aims to build a production-oriented Retrieval-Augmented Generation (RAG) system using:

* Dense vector retrieval
* BM25 keyword retrieval
* Reciprocal Rank Fusion
* Reranking
* Citation verification
* Confidence scoring
* Evaluation
* FastAPI
* Docker

## Project Status

Repository initialized. Implementation will be developed incrementally.

## Planned Architecture

Conceptual pipeline:

```
Documents
→ Ingestion
→ Chunking + Metadata
→ Dense Retrieval + BM25
→ Reciprocal Rank Fusion
→ Reranking
→ LLM Generation
→ Citation Verification
→ Confidence Scoring
→ Evaluation
→ FastAPI
→ Docker
```

## Planned Project Structure

* `app/`: Core application modules including ingestion, embeddings, retrieval, reranking, generation, citations, confidence, evaluation, and API routes.
* `data/`: Raw source documents, processed datasets, and evaluation test cases.
* `tests/`: Comprehensive unit and integration test suites.
* `scripts/`: Utility scripts for data setup, index generation, and running evaluations.
* `notebooks/`: Jupyter notebooks for exploratory analysis and experiments.
* `results/`: Output benchmarks, metric evaluations, and performance reports.
* `docs/`: Technical documentation and design specifications.

## Planned Development Phases

1. Project setup
2. Document ingestion
3. Chunking and metadata
4. Dense retrieval
5. BM25 retrieval
6. Reciprocal Rank Fusion
7. Reranking
8. LLM generation
9. Citation generation
10. Citation verification
11. Confidence scoring
12. Evaluation dataset and metrics
13. FastAPI
14. Testing
15. Docker
16. Benchmarking and optimization

## Important Engineering Goals

* **Reproducibility**: Deterministic data pipelines and repeatable evaluation setups.
* **Modular architecture**: Clear separation of concerns across ingestion, retrieval, reranking, and generation modules.
* **Measurable retrieval quality**: Continuous benchmarking using metrics like MRR, NDGG, and Hit Rate.
* **Citation traceability**: Grounding LLM responses directly in retrieved document sources.
* **Hallucination reduction**: Verifying claims against raw context to ensure high factual accuracy.
* **Evaluation-driven development**: Rigorous quantitative assessment at each pipeline component.
* **Production-oriented API design**: Robust RESTful service layer with clean error handling and streaming capabilities.

---

## Dataset: Nexora Synthetic Operations Corpus

The project uses the **Nexora RAG Evaluation Dataset** — a fully synthetic corpus of Nexora Smart Campus operational documents, designed for evaluating RAG pipelines.

### Counts

| Asset | Count |
|---|---|
| Source documents | 30 |
| Text chunks | 150 |
| Evaluation questions | 150 |

### Structure

```
data/
├── raw/
│   ├── documents/          # Individual JSON files (DOC001.json … DOC030.json)
│   └── documents.jsonl     # All 30 documents in a single JSONL file
├── processed/
│   └── chunks.jsonl        # 150 chunks (5 per document)
└── evaluation/
    └── questions.jsonl     # 150 evaluation questions
```

### Evaluation Question Categories

| Type | Count | Description |
|---|---|---|
| `direct` | 25 | Straightforward factual lookups |
| `semantic` | 25 | Questions requiring semantic understanding |
| `exact_keyword` | 25 | Questions containing exact phrases from the corpus |
| `multi_document` | 25 | Questions requiring evidence from multiple documents |
| `distractor` | 25 | Questions with potentially confusing surface-level matches |
| `unanswerable` | 25 | Questions the corpus cannot answer (tests hallucination resistance) |

### Validating the Dataset

Run the dataset validation script from the repository root:

```bash
python scripts/validate_dataset.py
```

A successful run prints:

```
========================================
NEXORA DATASET VALIDATION
========================================

Documents: 30/30 ✓
Chunks:    150/150 ✓
Questions: 150/150 ✓

Question distribution:
  direct          25 ✓
  semantic        25 ✓
  exact_keyword   25 ✓
  multi_document  25 ✓
  distractor      25 ✓
  unanswerable    25 ✓

Cross references:  PASS ✓
Duplicate IDs:     PASS ✓
Malformed records: PASS ✓

========================================
DATASET VALIDATION PASSED
========================================
```

Exit code `0` = all checks passed. Exit code `1` = validation failures. Exit code `2` = file not found.

---

## Phase 2: Dense Embeddings & Vector Retrieval

Dense vector retrieval maps textual queries and document chunks into a continuous high-dimensional vector space, allowing the system to retrieve relevant content based on **semantic meaning and context** rather than relying solely on exact keyword overlaps.

### Key Components

* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
  * Local, CPU-friendly inference (no external API calls required)
  * Output dimension: 384
  * L2-normalized embeddings
* **Vector Index**: FAISS `IndexFlatIP`
  * Inner product over L2-normalized vectors mathematically equals **Cosine Similarity**
  * Persistent storage under `data/vector_store/` (`index.faiss` + `metadata.json`)
* **Retrieval Service**: `DenseRetrievalService` in `app/retrieval/`
  * Validates query inputs and returns top-K `RetrievalResult` models with similarity scores and ranks

### Commands

**1. Build Vector Index**

Generate chunk embeddings and construct the local FAISS vector index:

```bash
python scripts/build_vector_index.py
```

**2. Evaluate Dense Retrieval**

Run dense vector retrieval against the 150 evaluation questions and compute Recall@1/3/5/10 and MRR:

```bash
python scripts/evaluate_dense_retrieval.py
```


