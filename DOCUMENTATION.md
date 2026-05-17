# DOCUMENTATION — semantic-rag-search

## 1. Overview

A retrieval-augmented generation (RAG) system for semantic search over scientific papers. Given a natural-language question, the system retrieves the most semantically relevant paper chunks from a FAISS index built on ~500 arXiv NLP abstracts, then generates a grounded answer via a local LLM (Ollama).

I built this to get hands-on with the full RAG pipeline — not just the embedding + vector search layer that most tutorials stop at, but also the chunking strategy comparison, cross-encoder reranking, and a disciplined evaluation setup with MRR and Hit@k metrics.

---

## 2. Architecture

```
arXiv API
   │
   ▼
src/corpus/downloader.py        — fetch & cache papers as JSONL
   │
   ▼
src/preprocessing/normalize.py  — strip LaTeX, collapse whitespace
src/preprocessing/chunker.py    — sentence / fixed-window / paragraph
   │
   ▼
src/embeddings/encoder.py       — batch encode via all-MiniLM-L6-v2
src/embeddings/index.py         — build FlatIP / IVFFlat FAISS index
   │
   ▼  (offline — saved to data/)
   │
   ▼  (query time)
src/retrieval/retriever.py      — encode query, search index, hydrate metadata
src/retrieval/reranker.py       — cross-encoder rerank (optional)
   │
   ▼
src/generation/generator.py     — RAG prompt + Ollama LLM call
   │
   ▼
GenerationResult (answer + source citations)
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/corpus/downloader.py` | Fetches arXiv abstracts via the `arxiv` library, caches as JSONL. Respects rate limits (3s delay). |
| `src/preprocessing/normalize.py` | Strips LaTeX commands, HTML tags, citation brackets, collapses whitespace. |
| `src/preprocessing/chunker.py` | Three chunking strategies (see §4). Produces `Chunk` dataclasses with full paper metadata. |
| `src/preprocessing/loader.py` | Orchestrates corpus loading → normalization → chunking → JSONL serialization. |
| `src/embeddings/encoder.py` | Wraps `SentenceTransformer` with batching, L2 normalization, and save/load helpers. |
| `src/embeddings/index.py` | Builds FlatIP and IVFFlat FAISS indices with a benchmark comparison and JSON manifest. |
| `src/retrieval/retriever.py` | Loads index + chunk store from disk, encodes queries, returns `SearchResult` objects. |
| `src/retrieval/reranker.py` | Cross-encoder reranking via `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `src/generation/generator.py` | Builds RAG prompt from retrieved chunks, calls Ollama, returns `GenerationResult`. |
| `src/evaluation/evaluator.py` | Synthetic QA generation, Hit@k, MRR, ROUGE-L evaluation. |

---

## 3. Data

**Source:** arXiv category `cs.CL` (Computation and Language), fetched via the arXiv API.

**Size:** ~500 papers (abstracts only — full PDFs not used).

**Schema:** Each paper stored as:
```json
{
  "paper_id": "2301.00001",
  "title": "...",
  "abstract": "...",
  "authors": ["..."],
  "categories": ["cs.CL"],
  "published": "2023-01-01",
  "url": "https://arxiv.org/abs/2301.00001"
}
```

**Preprocessing decisions:**
- LaTeX math expressions (`$...$`) are replaced with `<MATH>` placeholder rather than dropped, to preserve that a math expression existed in the context.
- `\textbf{X}` and similar formatting commands are unwrapped — the content matters, the markup doesn't.
- Citations like `[1,2]` are dropped entirely — they add noise without meaning in isolation.
- Unicode NFC normalization handles accented characters and ligatures from non-English authors.

---

## 4. Methodology

### 4.1 Chunking strategies

I compared three strategies to understand the precision/recall tradeoff at the chunk level.

| Strategy | Description | Avg words/chunk | Chunks per paper |
|---|---|---|---|
| `sentence` | Group 4 sentences per chunk with short-sentence merging | ~60 | 2–3 |
| `fixed_window` | 100-word sliding window, 50-word stride (50% overlap) | ~90 | 3–4 |
| `paragraph` | Title as chunk 0; each paragraph break starts a new chunk | ~120 | 1–2 |

**Why sentence chunking wins:** arXiv abstracts are dense and well-structured. Sentence-level chunks give tighter semantic focus, which improves cosine similarity alignment between a query and the retrieved chunk. The evaluation confirms this (MRR 0.733 for sentence vs. 0.642 for fixed-window vs. 0.497 for paragraph).

Paragraph strategy has high Hit@10 (0.98) but terrible Hit@1 (0.04) — the relevant chunk exists in the index but its broader text dilutes the embedding, pushing it down the ranking.

### 4.2 Embedding model

I used `sentence-transformers/all-MiniLM-L6-v2`:
- 384-dimensional, 22M parameters
- Designed for semantic similarity — trained on hundreds of millions of sentence pairs
- Runs fast on CPU (~200ms to encode 1k chunks on M1)
- Free, local, no API key

Vectors are L2-normalized before insertion so dot product equals cosine similarity.

### 4.3 Vector index

Two FAISS index types compared per strategy:

| Index | Build time | Query latency | IVF Recall@10 |
|---|---|---|---|
| FlatIP (exact) | <1ms | 0.01ms | 1.000 (ground truth) |
| IVFFlat (ANN) | ~5ms | 0.006ms | 0.928 |

At 1–2k vectors, FlatIP is fast enough that IVFFlat offers no meaningful speedup. I kept both in the codebase because IVFFlat matters at scale (10k+ vectors) and the comparison report is useful.

### 4.4 Retrieval pipeline

1. **Bi-encoder retrieval:** Encode query with the same all-MiniLM-L6-v2, search FlatIP index for top-k chunks.
2. **Optional cross-encoder reranking:** Pass (query, chunk) pairs through `cross-encoder/ms-marco-MiniLM-L-6-v2` to get a more accurate relevance score. Improves precision at the cost of O(k) forward passes.

Cross-encoder reranking is worth it when k ≤ 20 (manageable latency) and when the user wants the top result to be highly precise rather than top-10 recall.

### 4.5 RAG generation

The retrieved chunks are formatted into a numbered context block and passed to Ollama with a constrained prompt ("use only the provided context"). Temperature is set to 0.1 to minimize hallucination.

I chose Ollama over the Hugging Face Transformers approach because it's easier to swap models (single CLI command), has better memory management for Mac M-series chips, and the REST API is straightforward to call from Python.

---

## 5. Results

### Retrieval evaluation (150 synthetic QA pairs per strategy)

| Strategy | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|
| sentence | **0.520** | **0.967** | **0.973** | **0.993** | **0.733** |
| fixed_window | 0.407 | 0.893 | 0.947 | 0.980 | 0.642 |
| paragraph | 0.040 | 0.947 | 0.967 | 0.980 | 0.497 |

Synthetic QA pairs were generated from the corpus itself (chunk text → question), so these metrics measure retrieval self-consistency — how often the system retrieves the source chunk when asked about it.

### FAISS index comparison (see `reports/index_comparison_*.json`)

| Strategy | Flat latency (ms) | IVF latency (ms) | IVF Recall@10 |
|---|---|---|---|
| sentence | 0.010 | 0.006 | 0.928 |
| fixed_window | 0.012 | 0.025 | 0.912 |
| paragraph | 0.008 | 0.011 | 0.948 |

---

## 6. Tradeoffs & Decisions

**Sentence chunking over paragraph:** The paragraph strategy sounds intuitive but collapses an entire abstract into 1–2 chunks. The embedding of a 150-word chunk is a blurrier representation than a focused 60-word sentence group. MRR of 0.497 vs. 0.733 is the price.

**FlatIP over IVFFlat as default:** At this corpus size (~1k vectors), IVFFlat is sometimes slower than FlatIP because the Voronoi cell overhead outweighs the computation savings. Only switch to IVF if the corpus grows past ~10k chunks.

**all-MiniLM-L6-v2 over larger models:** I evaluated against the "use the biggest model" instinct. For short abstract-length text, MiniLM's 384d output is sufficient and the speed difference matters in interactive search. A 768d model like `all-mpnet-base-v2` would likely improve Hit@1 by a few points but add 3-4× encoding time.

**Synthetic QA for evaluation over human annotation:** Human annotation is expensive. Synthetic QA (treat a chunk as its own answer, ask the model to retrieve it) is a known lower bound — real retrieval is harder than this. The numbers are still useful for comparing strategies against each other.

**Not using a document chunker from a library (LangChain, LlamaIndex):** I wanted to understand exactly what chunking does to the embedding distribution, which means writing it from scratch. The implementation in `src/preprocessing/chunker.py` is 200 lines and fully transparent.

---

## 7. How to Run

### Setup

```bash
git clone https://github.com/AAdilCan/semantic-rag-search
cd semantic-rag-search
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

For LLM generation (optional):
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
ollama serve  # leave running in a separate terminal
```

### Build the corpus and index

```bash
# Download ~500 NLP papers from arXiv
python -m src.corpus.downloader --category cs.CL --max-papers 500

# Chunk and build FAISS indices for all strategies
python scripts/build_index.py --all-strategies
```

### Search

```bash
# Single query (sentence index, flat, no reranking)
python scripts/search.py --query "transformer attention mechanisms"

# Interactive mode
python scripts/search.py

# With cross-encoder reranking (fetches top-20, reranks to top-5)
python scripts/search.py --query "BERT fine-tuning NLP" --rerank --top-k 20 --rerank-top 5

# Different strategy
python scripts/search.py --strategy fixed_window --query "few-shot learning"
```

### Evaluate

```bash
# Retrieval metrics only (fast — no Ollama required)
python scripts/evaluate.py --no-generation --n-pairs 150

# Full RAG eval (requires Ollama running with llama3.2)
python scripts/evaluate.py --n-pairs 100 --gen-pairs 20

# Different strategy
python scripts/evaluate.py --strategy fixed_window --no-generation
```

### Run tests

```bash
pytest tests/ -v
# or with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 8. How to Extend

**Larger corpus:** Pass `--max-papers 5000` to the downloader and add more categories (`cs.LG`, `stat.ML`). IVFFlat becomes worthwhile at 10k+ chunks — switch with `--index-type ivf` in the search CLI.

**Better embedding model:** Swap `DEFAULT_MODEL` in `src/embeddings/encoder.py` to `sentence-transformers/all-mpnet-base-v2` (768d, higher quality) or a domain-specific model like `allenai/specter2` for scientific papers.

**Full papers instead of abstracts:** The `Paper` dataclass in `src/corpus/downloader.py` can be extended to fetch full PDFs via the arXiv API. The chunker would then produce hundreds of chunks per paper instead of 2–4.

**Different LLM:** Change `DEFAULT_MODEL` in `src/generation/generator.py` to any Ollama-supported model (`mistral`, `phi3`, `gemma2:2b`). The prompt template is model-agnostic.

**Human evaluation:** Replace synthetic QA in `src/evaluation/evaluator.py` with a curated `qa_pairs.jsonl` file and load that instead of generating synthetically.

---

## 9. References

- Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019) — basis for all-MiniLM-L6-v2
- Johnson et al., "Billion-scale similarity search with GPUs" (FAISS paper, 2019)
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — RAG formulation
- Nogueira & Cho, "Passage Re-ranking with BERT" (2019) — cross-encoder reranking approach
- arXiv API: https://info.arxiv.org/help/api/index.html
- sentence-transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss
- Ollama: https://ollama.com
