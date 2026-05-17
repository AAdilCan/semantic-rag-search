# semantic-rag-search

Retrieval-Augmented Generation system for semantic search over scientific papers. Retrieves relevant paper chunks from a FAISS index built on arXiv NLP abstracts and generates grounded answers via a local LLM — no API key required.

## Results

Evaluated on 150 synthetic QA pairs (chunk-as-answer retrieval):

| Strategy | Hit@1 | Hit@5 | MRR |
|---|---|---|---|
| sentence (best) | **0.520** | **0.973** | **0.733** |
| fixed_window | 0.407 | 0.947 | 0.642 |
| paragraph | 0.040 | 0.967 | 0.497 |

FAISS index comparison (sentence strategy): FlatIP 0.010ms/query vs. IVFFlat 0.006ms/query, IVF Recall@10 = 0.928.

## Architecture

```
arXiv API → Downloader → Normalizer → Chunker (3 strategies)
                                           ↓
                              all-MiniLM-L6-v2 Encoder
                                           ↓
                                  FAISS Vector Index
                                           ↓
User Query → Query Encoder → Top-K Retrieval → (Cross-Encoder Rerank)
                                           ↓
                              Ollama LLM → Grounded Answer
```

## Quick Start

```bash
git clone https://github.com/AAdilCan/semantic-rag-search
cd semantic-rag-search
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download ~500 NLP papers from arXiv
python -m src.corpus.downloader --category cs.CL --max-papers 500

# Build chunked JSONL + FAISS indices for all 3 strategies
python scripts/build_index.py --all-strategies

# Search
python scripts/search.py --query "transformer attention mechanisms"

# Interactive mode
python scripts/search.py

# With cross-encoder reranking
python scripts/search.py --query "few-shot NLP" --rerank --top-k 20
```

For RAG generation, install [Ollama](https://ollama.com/download) and run:
```bash
ollama pull llama3.2
ollama serve
python scripts/evaluate.py  # full eval with generation metrics
```

## Structure

```
src/
  corpus/        — arXiv fetching and JSONL caching
  preprocessing/ — chunking (sentence/fixed-window/paragraph) + normalization
  embeddings/    — sentence-transformer encoder, FAISS index builder
  retrieval/     — top-k retrieval, cross-encoder reranker
  generation/    — RAG prompt builder, Ollama LLM integration
  evaluation/    — Hit@k, MRR, ROUGE-L evaluation
scripts/
  build_index.py   — build embeddings + FAISS index from corpus
  search.py        — search CLI with rich output
  evaluate.py      — retrieval + generation evaluation
  explore_corpus.py — corpus statistics and figures
tests/           — pytest suite (30 tests)
data/
  processed/     — chunked JSONL files
  indices/       — FAISS index files + manifests
reports/         — evaluation JSON results
figures/         — EDA plots
```

## Dev

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for architecture details, chunking strategy analysis, methodology, and design decisions.
