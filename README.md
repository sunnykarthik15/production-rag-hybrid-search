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

---

## Phase 3: BM25 Lexical Retrieval

BM25 (Best Matching 25) is an inverse document frequency (IDF)-weighted probabilistic lexical retrieval algorithm. It scores relevance based on exact keyword occurrences, term frequencies, and document lengths.

### Key Differences & Why Both Dense and Lexical Are Needed

| Retrieval Technique | Core Mechanism | Strengths | Weaknesses |
|---|---|---|---|
| **Dense Vector Retrieval** | Continuous embedding space (`all-MiniLM-L6-v2`) | Captures **semantic similarity**, paraphrases, conceptual meaning, and multi-word intent. | Can fail on exact technical codes, serial numbers, rare codes, or specialized acronyms. |
| **BM25 Lexical Retrieval** | Probabilistic term matching (`rank-bm25`) | Exceptional at **exact keyword matching**, module identifiers (e.g. `NX-FAC-100`), numbers, and exact technical terminology. | Misses semantic synonyms or rephrased queries that do not share exact terms with documents. |

**Why Hybrid Search is Required:**
Dense retrieval excels when queries ask about general concepts ("operating schedule"), while BM25 excels when queries contain exact alphanumeric codes or rare terms ("NX-FAC-100"). Combining both via Reciprocal Rank Fusion (RRF) provides optimal recall across all query categories.

### Key Components

* **Lexical Index & Tokenization**: `BM25Store` in `app/retrieval/bm25_store.py`
  * Deterministic regex tokenization preserving numbers, terms, and hyphenated technical identifiers (e.g. `NX-FAC-100`).
  * Built using `rank-bm25` (`BM25Okapi`).
* **Retrieval Service**: `BM25RetrievalService` in `app/retrieval/bm25_service.py`
  * Provides `search(query, top_k)` returning structured `RetrievalResponse` models.

### Command

**Evaluate BM25 Retrieval**

Run BM25 retrieval against the 150 evaluation questions and compute Recall@1/3/5/10 and MRR:

```bash
python scripts/evaluate_bm25_retrieval.py
```

---

## Phase 4: Reciprocal Rank Fusion (RRF) & Hybrid Search

Phase 4 integrates Dense Vector Retrieval (`DenseRetrievalService`) and BM25 Lexical Retrieval (`BM25RetrievalService`) into a unified **Hybrid Search Pipeline** (`HybridRetrievalService`).

```
                              ┌────────────────────────┐
                              │    Input Query Text    │
                              └───────────┬────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │   Dense Vector Search     │               │   BM25 Lexical Search     │
      │  (all-MiniLM-L6-v2/FAISS) │               │   (rank-bm25 / Okapi)     │
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │ (candidate_k=50)                          │ (candidate_k=50)
                    ▼                                           ▼
            [ Dense Ranks ]                             [ BM25 Ranks ]
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Reciprocal Rank Fusion │
                              │     (rrf_k = 60)       │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │  Top-K Hybrid Results  │
                              └────────────────────────┘
```

### Reciprocal Rank Fusion (RRF) Algorithm

Reciprocal Rank Fusion (RRF) is an unsupervised rank aggregation method that combines multiple ranked result lists into a single unified output ranking.

#### RRF Formula

For each candidate text chunk $d$ in the candidate pool:

$$RRF\_Score(d) = \sum_{m \in \{Dense, BM25\}} \frac{1}{k_{rrf} + rank_m(d)}$$

Where:
- $rank_m(d)$ is the 1-based rank position of chunk $d$ in retrieval system $m$.
- $k_{rrf}$ is the smoothing hyperparameter (default = `60`).
- If chunk $d$ is missing from system $m$'s candidate list, its reciprocal rank contribution for that system is $0$.

#### Key Parameters

* **`candidate_k`** (default: `50`): The depth of candidate results fetched from both Dense and BM25 sub-services prior to fusion. Retrieving a deep candidate pool ensures relevant items ranked slightly lower in one sub-service are not prematurely lost before fusion.
* **`top_k`** (default: `10`): The final requested count of fused hybrid results returned to the caller.
* **`rrf_k`** (default: `60`): The smoothing constant balancing high-ranked versus lower-ranked candidates.

#### Why Raw Dense and BM25 Scores Are NOT Directly Combined

Raw similarity scores from Dense retrieval (Cosine Similarity in $[-1.0, 1.0]$) and BM25 retrieval (unbounded positive score in $[0.0, \infty)$) live on completely different mathematical scales and distributions. Directly summing or averaging raw scores would heavily bias results toward BM25 score magnitudes. RRF operates **purely on rank positions**, making it completely scale-invariant, robust against score distribution differences, and parameter-free regarding system weight tuning.

### Key Component

* **Hybrid Retrieval Service**: `HybridRetrievalService` in `app/retrieval/hybrid_service.py`
  * Public interface: `retrieve(query, top_k=10, rrf_k=60, candidate_k=50) -> RetrievalResponse`
  * Deterministic tie-breaking on `chunk_id` ascending.

### Command

**Evaluate Hybrid Retrieval**

Run hybrid retrieval against all 150 evaluation questions and compute Recall@1/3/5/10 and MRR:

```bash
python scripts/evaluate_hybrid_retrieval.py
```

### Benchmark Metric Results Across All Phases

| Retrieval Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| **Phase 2 — Dense Vector Search** | 0.1840 (18.40%) | 0.5520 (55.20%) | 0.6480 (64.80%) | 0.7840 (78.40%) | 0.3808 |
| **Phase 3 — BM25 Lexical Search** | 0.4160 (41.60%) | 0.7920 (79.20%) | 0.9200 (92.00%) | 1.0000 (100.00%) | 0.6209 |
| **Phase 4 — Hybrid Search (RRF)** | **0.2560 (25.60%)** | **0.6960 (69.60%)** | **0.8400 (84.00%)** | **0.9840 (98.40%)** | **0.5001** |
