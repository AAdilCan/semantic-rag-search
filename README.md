# semantic-rag-search

Retrieval-Augmented Generation (RAG) system for semantic search over scientific papers. Given a natural-language question, it retrieves the most relevant paper chunks from a FAISS index built on arXiv abstracts, then generates a grounded answer using an LLM.

## Results

| Metric | Score |
|---|---|
| Hit@1 | *TBD* |
| Hit@5 | *TBD* |
| MRR | *TBD* |
| ROUGE-L (answers) | *TBD* |

*Results updated after evaluation on Day 5.*

## Architecture

```
arXiv API → Corpus Downloader → Chunker → Sentence-Transformer Encoder
                                                        ↓
                                               FAISS Vector Index
                                                        ↓
User Query → Query Encoder → Top-K Retrieval → Cross-Encoder Reranker
                                                        ↓
                                            LLM Answer Generation
```

## Quick Start

```bash
git clone https://github.com/AAdilCan/semantic-rag-search
cd semantic-rag-search
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY

# Download corpus (~1k NLP papers from arXiv)
python -m src.corpus.downloader --category cs.CL --max-papers 1000

# Build FAISS index
python -m src.embeddings.encoder --input data/processed/chunks.jsonl

# Run a query
python -m src.pipeline query "What are the main approaches to few-shot learning in NLP?"
```

## Structure

```
src/
  corpus/       — arXiv fetching and caching
  preprocessing/ — chunking and text normalization
  embeddings/   — sentence-transformer encoding + FAISS indexing
  retrieval/    — top-k retrieval with optional reranking
  generation/   — LLM answer generation with context
  pipeline.py   — end-to-end RAG orchestration
scripts/
  explore_corpus.py  — corpus statistics and visualizations
tests/          — pytest suite
data/           — processed chunks and index (not raw downloads)
reports/        — evaluation results
figures/        — plots
```

## Development

```bash
pytest tests/ -v --cov=src
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for architecture details, methodology, and design decisions.
