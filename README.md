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
| **Phase 4 — Hybrid Search (RRF)** | 0.2560 (25.60%) | 0.6960 (69.60%) | 0.8400 (84.00%) | 0.9840 (98.40%) | 0.5001 |

---

## Phase 5: Cross-Encoder Reranking

Phase 5 introduces second-stage **Cross-Encoder Reranking** (`RerankerService` and `RerankedHybridService`) on top of Phase 4 Hybrid Search.

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
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Reciprocal Rank Fusion │
                              │   (RRF Candidate Pool) │
                              └───────────┬────────────┘
                                          │ (candidate_k=30)
                                          ▼
                              ┌────────────────────────┐
                              │ Cross-Encoder Reranker │
                              │ (ms-marco-MiniLM-L-6)  │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Final Top-K Results    │
                              └────────────────────────┘
```

### Architecture & Key Principles

1. **Two-Stage Pipeline**:
   - **Stage 1 (Retrieval)**: Dense Vector Search and BM25 Lexical Search independently retrieve candidates, fused via Reciprocal Rank Fusion (RRF) into a candidate pool (`candidate_k=30`).
   - **Stage 2 (Reranking)**: A Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) evaluates joint attention over pairs `(query, text)`, outputting deep relevance logit scores.
2. **Model Choice**:
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (6 transformer layers, ~22M parameters, ~80MB footprint).
   - CPU-friendly, fast inference sub-100ms per batch of 30 candidate pairs.
3. **Data Representation**:
   - Evaluates pair `(query, f"{title}: {text}")` to leverage document context.
   - Deterministic tie-breaking on `chunk_id` ascending when Cross-Encoder scores match.

### Key Components

* **Reranker Service**: `RerankerService` in `app/reranking/service.py`
  * Wraps `sentence_transformers.CrossEncoder`.
  * Public interface: `rerank(query, candidates, top_k=10) -> RetrievalResponse`
* **Reranked Hybrid Service**: `RerankedHybridService` in `app/reranking/hybrid_reranker.py`
  * Public interface: `retrieve(query, top_k=10, rrf_k=60, candidate_k=30) -> RetrievalResponse`

### Command

**Evaluate Reranked Hybrid Retrieval**

Run Reranked Hybrid search against all 150 evaluation questions and compute Recall@1/3/5/10 and MRR:

```bash
python scripts/evaluate_reranked_hybrid.py
```

### Complete Benchmark Comparison Across All Phases

| Retrieval Pipeline | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| **Phase 2 — Dense Vector Search** | 0.1840 (18.40%) | 0.5520 (55.20%) | 0.6480 (64.80%) | 0.7840 (78.40%) | 0.3808 |
| **Phase 3 — BM25 Lexical Search** | 0.4160 (41.60%) | 0.7920 (79.20%) | 0.9200 (92.00%) | 1.0000 (100.00%) | 0.6209 |
| **Phase 4 — Hybrid Search (RRF)** | 0.2560 (25.60%) | 0.6960 (69.60%) | 0.8400 (84.00%) | 0.9840 (98.40%) | 0.5001 |
| **Phase 5 — Reranked Hybrid (Cross-Encoder)** | **0.4720 (47.20%)** | **1.0000 (100.00%)** | **1.0000 (100.00%)** | **1.0000 (100.00%)** | **0.6920** |

---

## Phase 6: LLM Generation & RAG Orchestration

Phase 6 connects the Reranked Hybrid Search pipeline (`RerankedHybridService`) with a grounded LLM Generation Service (`GenerationService`) into an end-to-end RAG Orchestrator (`RAGService`).

```
                              ┌────────────────────────┐
                              │    User Query Text     │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Reranked Hybrid Search │
                              │ (Dense + BM25 + Cross) │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Top-K Citation Chunks  │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ Context Builder Prompt │
                              │   (prompts.py)         │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ LLM Provider           │
                              │ (Mock / OpenAI API)    │
                              └───────────┬────────────┘
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ RAG Response           │
                              │ (Grounded Answer       │
                              │  + Source Citations)   │
                              └────────────────────────┘
```

### Architecture & Grounded Generation Rules

1. **Vendor-Agnostic LLM Provider Abstraction (`LLMProvider`)**:
   - `MockLLMProvider`: Deterministic offline provider used for automated tests and offline benchmark evaluation.
   - `OpenAILLMProvider`: Live API provider supporting OpenAI / OpenAI-compatible REST endpoints via standard Python HTTP requests (zero mandatory external vendor SDK lock-in).
2. **Grounded Prompt Guidelines**:
   - Answers queries strictly using the provided context chunks.
   - Never hallucinates, assumes, or extrapolates facts outside the retrieved evidence.
   - Explicitly returns `"I do not have sufficient information in the provided context to answer this question."` when evidence is missing or context is empty.
3. **Structured Citation Response (`RAGResponse`)**:
   - Contains query, grounded answer text, and structured evidence citations (`RAGSource` objects with `chunk_id`, `document_id`, `title`, `department`, `rank`, and `score`).

### Key Components

* **Generation Models**: `GenerationRequest`, `GenerationResponse` in `app/generation/models.py`
* **Prompt Engineering**: `SYSTEM_PROMPT`, `format_context_block`, `build_grounded_prompt` in `app/generation/prompts.py`
* **Generation Service**: `GenerationService`, `LLMProvider`, `MockLLMProvider`, `OpenAILLMProvider` in `app/generation/service.py`
* **RAG Orchestrator**: `RAGService`, `RAGResponse`, `RAGSource` in `app/rag/service.py` and `app/rag/models.py`

### Configuration Settings

Set environment variables or edit `app/config.py`:

```bash
# Enable live OpenAI API (optional, defaults to offline mock provider)
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="your-api-key-here"
```

### Commands

**Run End-to-End RAG Benchmark Evaluation**

Evaluate the 150-question benchmark through the complete RAG orchestration pipeline:

```bash
python scripts/evaluate_rag.py
```

---

## Phase 7: Citation Verification & Answer Groundedness

Phase 7 introduces an automated post-generation **Citation Verification and Answer Groundedness** system (`app/citations/`), designed to evaluate claim faithfulness, citation accuracy, and attribution completeness without external LLM dependencies.

```
                               ┌────────────────────────┐
                               │    User Query Text     │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Reranked Hybrid Search │
                               │ (Dense + BM25 + Cross) │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Evidence Sufficiency   │
                               │ (min_relevance_score)  │
                               └───────────┬────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │ (Below Threshold)                         │ (Sufficient)
                     ▼                                           ▼
       ┌───────────────────────────┐               ┌───────────────────────────┐
       │ Return Insufficient-Info  │               │ Context Builder & Prompt  │
       │ Sentinel Answer           │               └─────────────┬─────────────┘
       └─────────────┬─────────────┘                             │
                     │                                           ▼
                     │                             ┌───────────────────────────┐
                     │                             │ LLM Generation            │
                     │                             └─────────────┬─────────────┘
                     │                                           │
                     └─────────────────────┬─────────────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ app/citations/ Layer   │
                               │                        │
                               │ 1. Refusal Detector    │
                               │ 2. Claim Segmenter     │
                               │ 3. Citation Parser     │
                               │ 4. Entity & Num Match  │
                               │ 5. Overlap & Entail    │
                               │ 6. Citation Correct    │
                               │ 7. Citation Complete   │
                               │ 8. Groundedness Calc   │
                               └───────────┬────────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                               │ Verified RAG Response  │
                               │ - answer_type          │
                               │ - refusal_correct      │
                               │ - claim_verifications  │
                               │ - groundedness_score   │
                               │ - citation_precision   │
                               │ - citation_recall / F1 │
                               │ - failure_diagnostics  │
                               └────────────────────────┘
```

### Core Architecture & Alignment Signals

1. **Decoupled Verification Concepts**:
   - **Claim Groundedness**: Is the factual assertion entailed/supported by evidence in the retrieved corpus?
   - **Citation Correctness**: Does the specific cited source chunk actually support the asserted claim?
   - **Citation Completeness**: Are all factual assertions attributed with source citation references?
2. **Multi-Signal Alignment Engine (`ClaimVerifier`)**:
   - **Entity / Code Alignment**: Direct extraction of operational codes (e.g. `NX-FAC-100`), proper nouns, and entity names.
   - **Temporal / Date Alignment**: Extraction of dates, calendar months, weekdays (e.g. *Wednesday* vs *Friday*), and clock times.
   - **Numerical Alignment**: Currency values, percentages, and quantities (e.g. *$80* vs *$50*).
   - **Polarity / Negation Detection**: Detection of negation modifiers and antonym inversions (*restricted* vs *unrestricted*).
   - **Directional Content-Token Containment & LCS Overlap**: Token precision and Longest Common Subsequence ratio against candidate chunks.
3. **Robust Unanswerable & Refusal Accounting**:
   - Sentinel refusal answers produce: `answer_type = "insufficient_evidence"`, `refusal_correct = True`, `groundedness_score = 1.0`, `total_claims = 0`, `spurious_citations = 0`.
   - Fabricated answers to unanswerable queries are flagged as `answer_type = "fabricated_answer"`, `refusal_correct = False`, `groundedness_score = 0.0`.

### Key Components

* **Data Models**: `AnswerType`, `UnsupportedReason`, `Claim`, `ClaimVerification`, `CitationVerificationResult` in `app/citations/models.py`
* **Extraction Engine**: `extract_claims`, `parse_citations_from_text`, `extract_entities`, `split_sentences` in `app/citations/extractor.py`
* **Verification Engine**: `ClaimVerifier`, `tokenize_content_words`, `compute_lcs_length` in `app/citations/verifier.py`
* **Verification Service**: `CitationVerificationService` in `app/citations/service.py`
* **Orchestration**: `RAGService` in `app/rag/service.py` integrating post-generation verification into `RAGResponse`.

### Commands

**1. Run Citation Verification Benchmark Evaluation**

Run end-to-end verification and metric distribution profiling across all 150 benchmark questions:

```bash
python scripts/evaluate_citation_verification.py
```

---

## Phase 7.2: Adversarial Hardening & Production-Readiness Pass

Phase 7.2 hardens the citation verification engine against evaluation leakage by testing across 20 distinct adversarial perturbation categories (A through T), calibrating support overlap thresholds across a full parameter grid, and enabling flexible live vs mock LLM evaluation.

### Key Audit Capabilities & Enhancements

1. **Adversarial Perturbation Suite (Categories A through T)**:
   - Evaluates benign paraphrases, exact citations, wrong source citations, missing citations, invalid indices, fabricated numbers, dates, weekdays, clock times, polarity inversions, negation insertions/removals, unsupported entities, unsupported claims, partial support, distractor citations, and contradictory sources.
2. **Support Threshold Calibration Grid**:
   - Sweeps overlap thresholds from `0.20` to `0.70` across benign and adversarial samples to empirically determine the optimal production balance.
3. **MockLLMProvider Simulation Engine**:
   - Dynamic deterministic perturbation engine supporting 15 simulation modes (`faithful_paraphrase`, `numeric_mutation`, `temporal_mutation`, `polarity_inversion`, `negation_insertion`, `negation_removal`, `missing_citations`, `wrong_citation`, etc.).
4. **Live LLM Evaluation Runner (`scripts/evaluate_live_rag.py`)**:
   - Decoupled CLI supporting live OpenAI APIs with automatic key validation, secret redaction, and deterministic fallback.

---

### Adversarial Evaluation Results (Categories A–T)

```
===========================================================================
PHASE 7.2 ADVERSARIAL CITATION VERIFICATION BENCHMARK
===========================================================================

Confusion Matrix:
                      Actual Supported    Actual Unsupported
  Pred Supported      TP: 3               FP: 0
  Pred Unsupported    FN: 0               TN: 17

Verification Performance Metrics:
  Total Test Cases:            20
  Accuracy:                    1.0000 (100.00%)
  Precision:                   1.0000 (100.00%)
  Recall:                      1.0000 (100.00%)
  F1 Score:                    1.0000
  False Positive Rate (FPR):   0.0000 (0.00%)
  False Negative Rate (FNR):   0.0000 (0.00%)
  Paraphrase Acceptance Rate:  1.0000 (100.00%)
  Adversarial Rejection Rate:  1.0000 (100.00%)
```

#### Category Breakdown (A through T)

| Cat | Perturbation Category | Expected Grounded | Expected Citation Correct | Audit Status | Primary Failure Classification |
|---|---|---|---|---|---|
| **A** | Faithful paraphrase | True | True | **PASS** | Valid support |
| **B** | Exact source statement | True | True | **PASS** | Valid support |
| **C** | Missing citation | True | False | **PASS** | `missing_citation` |
| **D** | Wrong citation | False | False | **PASS** | `wrong_citation` |
| **E** | Invalid citation index | False | False | **PASS** | `invalid_citation_index` |
| **F** | Fabricated number | False | False | **PASS** | `numerical_mismatch` |
| **G** | Fabricated date | False | False | **PASS** | `temporal_mismatch` |
| **H** | Fabricated weekday | False | False | **PASS** | `temporal_mismatch` |
| **I** | Fabricated time | False | False | **PASS** | `temporal_mismatch` |
| **J** | Polarity inversion | False | False | **PASS** | `contradiction` |
| **K** | Negation insertion | False | False | **PASS** | `contradiction` |
| **L** | Negation removal | False | False | **PASS** | `contradiction` |
| **M** | Unsupported entity | False | False | **PASS** | `entity_mismatch` |
| **N** | Unsupported claim | False | False | **PASS** | `no_supporting_source` |
| **O** | Partial claim | False | False | **PASS** | `no_supporting_source` |
| **P** | Multi-source correct attribution | True | True | **PASS** | Valid support |
| **Q** | Multi-source wrong attribution | False | False | **PASS** | `wrong_citation` |
| **R** | Distractor citation | True | False | **PASS** | `wrong_citation` |
| **S** | Contradictory numerical value | False | False | **PASS** | `numerical_mismatch` |
| **T** | Contradictory temporal value | False | False | **PASS** | `temporal_mismatch` |

---

### Support Threshold Calibration Grid

| Overlap Threshold | Accuracy | Precision | Recall | F1 Score | FPR | FNR | Paraphrase Acceptance | Adversarial Rejection |
|---|---|---|---|---|---|---|---|---|
| 0.20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.25 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.35 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| **0.40 (Recommended)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.0000** | **0.0000** | **100.00%** | **100.00%** |
| 0.45 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.55 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.65 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 100.00% | 100.00% |
| 0.70 | 0.9500 | 1.0000 | 0.6667 | 0.8000 | 0.0000 | 0.3333 | 66.67% | 100.00% |

* **Threshold Plateau**: Overlap thresholds between `0.20` and `0.65` yield 100% F1.
* **Degradation Point**: At threshold `0.70`, recall falls to 66.67% due to false rejection of legitimate benign paraphrases.
* **Production Recommendation**: `CITATION_MIN_OVERLAP_THRESHOLD = 0.40` provides an optimal balance between lexical variation tolerance and hallucination prevention.

---

### Verifier Boundary & Production Limitations

1. **Deterministic Speed**: The verifier operates in <5ms per claim without external LLM inference overhead.
2. **Deterministic Coverage**: Guarantees zero tolerance on altered alphanumeric codes, temporal indicators (weekdays, dates, times), and numeric/financial quantities.
3. **Inference Limitations**: Lexical LCS and token containment do not perform multi-hop symbolic arithmetic or complex deduction across non-adjacent clauses. High-consequence regulatory domains should pair this with an asynchronous NLI Cross-Encoder or second-stage LLM-as-judge.

---

### Commands & Execution

**1. Run Adversarial Citation Benchmark**
```bash
python scripts/evaluate_adversarial_citations.py
```

**2. Run Threshold Calibration**
```bash
python scripts/calibrate_thresholds.py
```

**3. Run Offline / Simulation Evaluation**
```bash
# Evaluate with faithful paraphrase mode:
python scripts/evaluate_live_rag.py --provider mock --mock-mode faithful_paraphrase

# Evaluate with adversarial numeric mutation:
python scripts/evaluate_live_rag.py --provider mock --mock-mode numeric_mutation
```

**4. Run Live LLM Evaluation (Requires OPENAI_API_KEY)**
```bash
python scripts/evaluate_live_rag.py --provider openai --model gpt-3.5-turbo
```

**5. Run Full Test Suite**
```bash
python -m pytest
```


---

## Phase 8: Production API & Application Layer

### Architecture

Phase 8 exposes the existing RAG pipeline through a production-oriented FastAPI application without duplicating any retrieval, generation, or verification logic.

```
                    HTTP Client (cURL / Browser / Frontend)
                                     │
                                     ▼
                          FastAPI App (app/main.py)
                          ├─ CORS (explicit origins)
                          ├─ Security Headers Middleware
                          ├─ Structured Logging
                          └─ Sanitized Exception Handlers
                                     │
                                     ▼
                           API Router (app/api/routes.py)
                   ┌─────────────────┼─────────────────┐
                   │                 │                 │
                   ▼                 ▼                 ▼
             GET /health       POST /query       POST /retrieve
             (readiness)      (full RAG)        (debug retrieval)
                   │                 │                 │
                   │                 ▼                 ▼
                   │         RAGService.answer()  RerankedHybridService
                   │         └→ Retrieval         .retrieve()
                   │         └→ Generation
                   │         └→ Verification
                   └─────────────────┼─────────────────┘
                                     │
                                     ▼
                            Structured JSON Response
```

### Endpoints

#### GET /health

Returns application and component readiness status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "environment": "mock",
  "components": {
    "retrieval": true,
    "reranker": true,
    "generation": true,
    "citation_verifier": true
  }
}
```

Status values: `"healthy"` (all components ready), `"degraded"` (one or more components unavailable).

Environment values: `"mock"` (offline MockLLMProvider), `"live"` (OpenAI API with valid key).

#### POST /query

Full RAG pipeline: retrieval → generation → citation verification.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the application deadline?", "top_k": 5}'
```

Request body:
| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `query` | string | Yes | 1–2000 chars, trimmed, non-empty |
| `top_k` | integer | No (default: 10) | 1–150 |

Response:
```json
{
  "query": "What is the application deadline?",
  "answer": "Based on the provided documentation: ...",
  "sources": [
    {
      "chunk_id": "DOC001_CHUNK_01",
      "document_id": "DOC001",
      "title": "...",
      "department": "...",
      "rank": 1,
      "score": 0.95,
      "text": "..."
    }
  ],
  "verification": { "...": "..." },
  "groundedness_score": 1.0,
  "metadata": {
    "retrieval_latency_ms": null,
    "generation_latency_ms": null,
    "verification_latency_ms": null,
    "total_latency_ms": 592.4,
    "llm_provider": "mock",
    "is_live_llm": false
  }
}
```

#### POST /retrieve

Debug/inspection endpoint. Retrieval only — no generation, no verification.

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "application deadline", "top_k": 5}'
```

Response:
```json
{
  "query": "application deadline",
  "top_k": 5,
  "results": [
    {
      "chunk_id": "DOC001_CHUNK_01",
      "document_id": "DOC001",
      "title": "...",
      "department": "...",
      "rank": 1,
      "score": 0.85,
      "text": "..."
    }
  ]
}
```

### Validation Rules

| Rule | HTTP Status | Error Code |
|------|-------------|------------|
| Empty or whitespace-only query | 422 | `VALIDATION_ERROR` |
| Query exceeds 2000 characters | 422 | `VALIDATION_ERROR` |
| `top_k` < 1 or > 150 | 422 | `VALIDATION_ERROR` |
| RAG pipeline failure | 500 | `RAG_ERROR` |
| Unexpected server error | 500 | `INTERNAL_ERROR` |

### Error Format

All errors return a structured envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": "body -> query: String should have at least 1 character"
  }
}
```

Stack traces, API keys, and filesystem paths are never exposed to clients.

### Local Startup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (development mode with auto-reload)
uvicorn app.main:app --reload

# Start the API server (production mode)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Swagger / OpenAPI Documentation

Interactive API documentation is automatically available:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### Docker

```bash
# Build the image
docker build -t nexora-rag-api .

# Run the container
docker run -p 8000:8000 nexora-rag-api

# Run with live OpenAI provider
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here nexora-rag-api
```

### Environment Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `LLM_PROVIDER` | `mock` | LLM backend (`mock` or `openai`) |
| `OPENAI_API_KEY` | *(none)* | Required only for live OpenAI provider |

### Mock vs Live Provider

| Condition | Environment | is_live_llm |
|-----------|-------------|-------------|
| No `OPENAI_API_KEY` set | `"mock"` | `false` |
| `LLM_PROVIDER=mock` | `"mock"` | `false` |
| `LLM_PROVIDER=openai` + valid key | `"live"` | `true` |

### Latency Metadata

All `/query` responses include monotonic timing metadata. `total_latency_ms` measures the complete RAG pipeline execution time using `time.monotonic()` (not wall-clock time).

### Security Considerations

- API keys are **never** logged, printed, or returned in responses.
- Error responses are sanitized to prevent leakage of secrets or filesystem paths.
- Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, `Referrer-Policy`, `Cache-Control`) are set on all responses.
- CORS is explicitly configured (not wildcard).
- Query length is validated at the API boundary (max 2000 characters).

### Frozen Retrieval Baseline

Phase 5 retrieval metrics remain exactly unchanged through Phase 8:

| Metric | Value |
|--------|-------|
| Recall@1 | 0.4720 |
| Recall@3 | 1.0000 |
| Recall@5 | 1.0000 |
| Recall@10 | 1.0000 |
| MRR | 0.6920 |

### Production Readiness Status

Phase 7.2 verdict: **READY_FOR_LIMITED_PRODUCTION**

Phase 8 adds operational API exposure, structured error handling, and container deployment capability. The underlying RAG verification remains deterministic and lexical/rule-based; it does not perform symbolic multi-hop reasoning or advanced semantic NLI. Do not claim this system is universally hallucination-proof.

---

## Phase 8.1: Production API Audit & Hardening Pass

### Audit Summary & Hardening Passes

Phase 8.1 performed a comprehensive security, lifecycle, error-handling, and concurrency audit of the API layer:

1. **HTTP Exception Envelopes**: Integrated `StarletteHTTPException` handler in `app/api/errors.py` so standard HTTP client errors (404 Not Found, 405 Method Not Allowed) return structured JSON envelopes rather than being caught and masked as generic 500 Internal Errors.
2. **Cross-Platform Path Sanitization**: Hardened `_PATH_PATTERN` regex in `app/api/errors.py` to sanitize both Windows backslash (`F:\...`), forward-slash (`C:/...`), and POSIX (`/home/...`) filesystem paths from error messages.
3. **Singleton Model Sharing**: Refactored `get_hybrid_reranker()` in `app/api/dependencies.py` to share the underlying `RerankedHybridService` from `RAGService`, eliminating duplicate Cross-Encoder model and FAISS vector index loading in memory.
4. **Health Check Accuracy**: Updated `check_component_health()` in `app/api/dependencies.py` to verify actual `is_initialized` states of `VectorStore`, `BM25Store`, and `CrossEncoder`, returning `status="degraded"` when stores are uninitialized.
5. **Configurable CORS & Environment Overrides**: Centralized environment overrides for `API_PORT`, `API_HOST`, `API_MAX_QUERY_LENGTH`, and `API_CORS_ORIGINS` in `app/config.py`.
6. **Concurrency & Thread Safety**: Verified multi-threaded request isolation across concurrent `/query` and `/retrieve` calls with zero cross-request state contamination.

### Audit Runner

Run the automated audit suite:

```bash
python scripts/audit_phase8.py
```

Generated audit artifact: `results/phase8_production_audit.json`

### Phase 8.1 Test Suite Status

- **Baseline Tests (Phases 1–7.2)**: 246 passed
- **Phase 8 API Unit Tests**: 21 passed
- **Phase 8.1 Hardening Tests**: 17 passed
- **Total Tests**: **284 passed, 0 failed**

### Final Production Verdict

**READY_FOR_LIMITED_PRODUCTION**

---

## Phase 9: Production Observability, Evaluation & Deployment Hardening

Phase 9 transitions the RAG system from a verified, hardened API into a measurable, observable, reproducible, deployable, and continuously evaluable production service.

```
                      INCOMING HTTP REQUEST
                                │
                                ▼
               ┌──────────────────────────────────┐
               │    ObservabilityMiddleware       │
               │  - X-Request-ID Correlation      │
               │  - SHA-256 Query Hashing (PII)   │
               │  - Structured JSON Access Log    │
               │  - Monotonic Latency Tracking    │
               └────────────────┬─────────────────┘
                                │
                                ▼
                       FastAPI Router
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
      GET /health           GET /ready          GET /metrics
     (Liveness 200)      (Readiness 200/503)   (Counters & Percentiles)
            │                   │                   │
            └───────────────────┼───────────────────┘
                                ▼
                      POST /query, /retrieve
                                │
                                ▼
                 RAG Engine with Sub-Phase Metrics
                (Retrieval → Rerank → Gen → Verify)
```

### Key Components

#### 1. Structured JSON Logging & Privacy Protection (`app/observability/logging.py`)
* **Deterministic Query Hashing**: User queries are never logged in plaintext. They are hashed using SHA-256 (`get_query_hash()`) to enable correlation while preventing PII exposure.
* **Recursive Redaction**: `sanitize_log_data()` automatically strips OpenAI API keys (`sk-...`), Bearer authorization tokens, and both Windows (`C:\...`, `F:/...`) and POSIX paths.
* **Structured Format**: `StructuredJSONFormatter` emits standard JSON logs with timestamp, log level, event name, component, request ID, duration, and sanitized contextual metadata.

#### 2. Thread-Safe In-Memory Metrics Collector (`app/observability/metrics.py`)
* **Singleton Architecture**: `MetricsCollector` maintains thread-safe request counters, error rates, and latency sample arrays protected by reentrant threading locks.
* **Percentile Distributions**: Accurately computes Mean, P50, P90, P95, and P99 latency distributions with zero-division safety across all endpoints and internal RAG sub-phases (`retrieval_ms`, `reranking_ms`, `generation_ms`, `verification_ms`).
* **Metrics Snapshot**: Exposed via `GET /metrics` for operational dashboards and health scraping.

#### 3. Correlation Middleware (`app/observability/middleware.py`)
* **`X-Request-ID` Lifecycle**: Ingests incoming `X-Request-ID` headers or generates compliant UUIDv4 identifiers. Replaces malformed or oversized IDs (>64 chars) with safe UUIDs.
* **Context Propagation**: Injects `request_id` into response headers and attaches it to all structured log records.

#### 4. Kubernetes Liveness & Readiness Probes (`app/api/routes.py`)
* **`GET /health`**: High-frequency liveness probe returning HTTP 200 with service status, environment mode, and uptime.
* **`GET /ready`**: Deep readiness probe verifying the initialization of `VectorStore` (FAISS), `BM25Store`, and `CrossEncoder`. Returns HTTP 200 when traffic-ready, or HTTP 503 Service Unavailable when stores are uninitialized or degraded.

#### 5. Hardened Non-Root Containerization (`Dockerfile`)
* **Security Lockdown**: Runs as non-root user `appuser` (UID 10001) in group `appuser`.
* **Ownership Enforcement**: Explicitly chowns `/app` to `appuser:appuser` to prevent privilege escalation.
* **Native Healthcheck**: Built-in `HEALTHCHECK` periodically probing `http://localhost:8000/health`.

---

### Phase 9 API Endpoints

#### GET /health (Liveness)
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "environment": "mock",
  "is_live_llm": false,
  "components": {
    "retrieval": true,
    "reranker": true,
    "generation": true,
    "citation_verifier": true
  }
}
```

#### GET /ready (Readiness)
```bash
curl http://localhost:8000/ready
```
```json
{
  "status": "ready",
  "ready": true,
  "checks": {
    "vector_store": true,
    "bm25_store": true,
    "cross_encoder": true
  }
}
```

#### GET /metrics (Operational Telemetry)
```bash
curl http://localhost:8000/metrics
```
```json
{
  "uptime_seconds": 124.5,
  "requests": {
    "total": 50,
    "by_endpoint": { "/health": 10, "/ready": 10, "/query": 30 },
    "errors": { "total": 0, "4xx": 0, "5xx": 0, "rate": 0.0 }
  },
  "latency_ms": {
    "overall": { "count": 50, "mean": 12.4, "p50": 1.2, "p90": 28.5, "p95": 32.1, "p99": 35.0 },
    "by_endpoint": { ... }
  },
  "rag": {
    "queries_processed": 30,
    "refusals": 5,
    "grounded_answers": 25,
    "groundedness_rate": 1.0,
    "subphase_latency_ms": {
      "retrieval": { "p50": 8.1, "p95": 14.2 },
      "reranking": { "p50": 12.3, "p95": 18.0 },
      "generation": { "p50": 0.5, "p95": 0.8 },
      "verification": { "p50": 2.1, "p95": 3.4 }
    }
  }
}
```

---

### Full Orchestrated Evaluation & Benchmarks

Run the complete, reproducible Phase 9 evaluation suite:

```bash
python scripts/run_full_evaluation.py
```

Run standalone API performance benchmarking:

```bash
python scripts/evaluate_api_performance.py --requests 50 --concurrency 5
```

Generated evaluation artifacts:
* `results/full_evaluation_report.json` — Comprehensive multi-phase quantitative telemetry.
* `results/phase9_production_readiness.json` — Categorical pass/fail production readiness scorecard.
* `results/api_performance_eval.json` — Latency percentiles and throughput benchmarks across all endpoints.

---

### Frozen Retrieval Baseline Verification

Phase 5 retrieval metrics remain strictly preserved across all phases:

| Metric | Phase 5 Baseline | Phase 9 Measured | Verification Status |
|---|---|---|---|
| **Recall@1** | `0.4720` | `0.4720` | Exact Match ✓ |
| **Recall@3** | `1.0000` | `1.0000` | Exact Match ✓ |
| **Recall@5** | `1.0000` | `1.0000` | Exact Match ✓ |
| **Recall@10** | `1.0000` | `1.0000` | Exact Match ✓ |
| **MRR** | `0.6920` | `0.6920` | Exact Match ✓ |

---

### Complete Test Suite Status

```
============================== test session starts ==============================
collected 328 items

tests/integration/ ... 23 passed
tests/unit/ ... 305 passed

====================== 328 passed, 0 failed in 177.63s =======================
```

* **Baseline Unit & Integration Tests (Phases 1–7.2)**: 246 passed
* **Phase 8 & 8.1 API Hardening Tests**: 38 passed
* **Phase 9 Observability, Metrics & Health Tests**: 44 passed
* **Total Automated Tests**: **328 passed, 0 failed (100% pass rate)**

---

### Production Readiness Verdict & Guardrails

**Overall System Status**: `READY_FOR_LIMITED_PRODUCTION`

**System Strengths**:
* Deterministic, zero-network mock generation for reproducible offline evaluations.
* Lexical, token-precision, and entity-preserving citation verification engine.
* Low latency (<30ms P95 query response time with local models) and high throughput (>45 req/sec).
* Comprehensive observability with privacy-preserving SHA-256 query logging and secret redaction.
* Hardened non-root Docker deployment container with liveness/readiness orchestration probes.

**Known Limitations & Guardrails**:
* Verification is token-precision, entity, and sentence-boundary based; it does not perform multi-step symbolic reasoning.
* Live LLM mode requires an external OpenAI API key and relies on network connectivity and upstream rate limits.
* FAISS and BM25 indexes are stored locally on the container filesystem; horizontal autoscaling requires shared volume storage or a distributed vector database.

