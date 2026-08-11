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
