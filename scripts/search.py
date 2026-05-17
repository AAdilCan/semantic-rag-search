"""
Interactive semantic search CLI over arXiv papers.

Usage:
    # Single query
    python scripts/search.py --query "transformer attention mechanisms"

    # Interactive mode
    python scripts/search.py

    # With cross-encoder reranking
    python scripts/search.py --query "BERT fine-tuning" --rerank --top-k 20

    # Different chunking strategy
    python scripts/search.py --query "few-shot learning" --strategy paragraph
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.retriever import Retriever

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.option("--query", "-q", default=None, help="Search query. Omit for interactive mode.")
@click.option("--top-k", "-k", default=10, show_default=True, help="Results to retrieve.")
@click.option("--rerank", is_flag=True, default=False, help="Apply cross-encoder reranking.")
@click.option("--rerank-top", default=5, show_default=True, help="Results to keep after reranking.")
@click.option(
    "--strategy",
    default="sentence",
    show_default=True,
    type=click.Choice(["sentence", "fixed_window", "paragraph"]),
    help="Chunking strategy index to search.",
)
@click.option(
    "--index-type",
    default="flat",
    show_default=True,
    type=click.Choice(["flat", "ivf"]),
    help="FAISS index type.",
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    query: Optional[str],
    top_k: int,
    rerank: bool,
    rerank_top: int,
    strategy: str,
    index_type: str,
    verbose: bool,
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    chunks_path = Path(f"data/processed/cs_CL_papers_{strategy}.jsonl")
    index_dir = Path(f"data/indices/{strategy}/{index_type}")

    if not chunks_path.exists():
        console.print(f"[red]Chunks not found: {chunks_path}[/red]")
        console.print("Run: python scripts/build_index.py --all-strategies")
        sys.exit(1)

    with console.status("Loading retriever…"):
        retriever = Retriever.from_disk(index_dir, chunks_path)

    reranker = None
    if rerank:
        from src.retrieval.reranker import CrossEncoderReranker
        with console.status("Loading cross-encoder reranker…"):
            reranker = CrossEncoderReranker()

    if query:
        _do_search(retriever, reranker, query, top_k, rerank_top)
    else:
        console.print(
            Panel(
                "[bold cyan]Semantic RAG Search[/bold cyan]\n"
                "Type a query and press Enter. Ctrl+C to quit.",
                subtitle=f"strategy={strategy} · index={index_type}",
            )
        )
        while True:
            try:
                q = console.input("[bold green]Query:[/bold green] ").strip()
                if not q:
                    continue
                _do_search(retriever, reranker, q, top_k, rerank_top)
            except (KeyboardInterrupt, EOFError):
                console.print("\nBye!")
                break


def _do_search(retriever, reranker, query: str, top_k: int, rerank_top: int) -> None:
    results = retriever.search(query, top_k=top_k)

    if reranker and results:
        results = reranker.rerank(query, results, top_n=rerank_top)
        label = f"Top {len(results)} (reranked)"
    else:
        label = f"Top {len(results)}"

    table = Table(
        title=f"[bold]{label}[/bold] for: [italic]{query}[/italic]",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Title", style="cyan", ratio=2)
    table.add_column("Chunk", ratio=3)
    table.add_column("Authors", ratio=1, style="dim")

    for r in results:
        authors_str = ", ".join(r.authors[:2])
        if len(r.authors) > 2:
            authors_str += " et al."
        snippet = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            snippet += "…"
        table.add_row(
            str(r.rank),
            f"{r.score:.3f}",
            r.title,
            snippet,
            authors_str,
        )

    console.print(table)
    console.print()


if __name__ == "__main__":
    main()
