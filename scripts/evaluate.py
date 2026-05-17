"""
Evaluation script: retrieval metrics (MRR, Hit@k) + optional generation ROUGE-L.

Usage:
    # Retrieval-only eval (no LLM required, fast)
    python scripts/evaluate.py --no-generation --n-pairs 200

    # Full RAG eval (requires `ollama serve` + `ollama pull llama3.2`)
    python scripts/evaluate.py --n-pairs 100 --gen-pairs 30

    # Different chunking strategy
    python scripts/evaluate.py --strategy fixed_window --no-generation
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import evaluate_generation, evaluate_retrieval, generate_synthetic_qa
from src.retrieval.retriever import Retriever

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--strategy",
    default="sentence",
    show_default=True,
    type=click.Choice(["sentence", "fixed_window", "paragraph"]),
)
@click.option("--n-pairs", default=100, show_default=True, help="Synthetic QA pairs to generate.")
@click.option("--top-k", default=10, show_default=True, help="Retrieval depth for eval.")
@click.option("--no-generation", is_flag=True, default=False, help="Skip LLM generation eval.")
@click.option("--gen-pairs", default=20, show_default=True, help="Pairs used for generation eval.")
@click.option("--ollama-model", default="llama3.2", show_default=True)
@click.option(
    "--output",
    default="reports/evaluation_results.json",
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.option("--verbose", "-v", is_flag=True, default=False)
def main(
    strategy: str,
    n_pairs: int,
    top_k: int,
    no_generation: bool,
    gen_pairs: int,
    ollama_model: str,
    output: Path,
    verbose: bool,
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    chunks_path = Path(f"data/processed/cs_CL_papers_{strategy}.jsonl")
    index_dir = Path(f"data/indices/{strategy}/flat")

    if not chunks_path.exists():
        console.print(f"[red]No chunks at {chunks_path}. Run scripts/build_index.py first.[/red]")
        sys.exit(1)

    with console.status("Loading retriever…"):
        retriever = Retriever.from_disk(index_dir, chunks_path)

    with console.status(f"Generating {n_pairs} synthetic QA pairs…"):
        qa_pairs = generate_synthetic_qa(chunks_path, n_pairs=n_pairs)

    console.print(f"Generated [bold]{len(qa_pairs)}[/bold] QA pairs")

    with console.status(f"Running retrieval evaluation (top_k={top_k})…"):
        ret_metrics = evaluate_retrieval(retriever, qa_pairs, top_k=top_k)

    _print_retrieval_table(ret_metrics, strategy)

    results: dict = {
        "strategy": strategy,
        "n_pairs": len(qa_pairs),
        "top_k": top_k,
        "retrieval": ret_metrics.to_dict(),
    }

    if not no_generation:
        try:
            from src.generation.generator import RAGGenerator

            with console.status(f"Loading Ollama ({ollama_model}) for generation eval…"):
                generator = RAGGenerator(model=ollama_model)
                gen_metrics = evaluate_generation(
                    generator, retriever, qa_pairs, top_k=5, max_pairs=gen_pairs
                )
            _print_generation_table(gen_metrics)
            results["generation"] = gen_metrics.to_dict()
        except RuntimeError as e:
            console.print(f"[yellow]Generation eval skipped:[/yellow] {e}")
            console.print("[dim]To enable: run 'ollama serve' and 'ollama pull llama3.2'[/dim]")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    console.print(f"\nResults saved → [bold]{output}[/bold]")


def _print_retrieval_table(metrics, strategy: str) -> None:
    table = Table(title=f"Retrieval Metrics  (strategy: {strategy})", box=box.SIMPLE_HEAD)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")

    table.add_row("Hit@1", f"{metrics.hit_at_1:.4f}")
    table.add_row("Hit@3", f"{metrics.hit_at_3:.4f}")
    table.add_row("Hit@5", f"{metrics.hit_at_5:.4f}")
    table.add_row("Hit@10", f"{metrics.hit_at_10:.4f}")
    table.add_row("MRR", f"{metrics.mrr:.4f}")
    table.add_row("Queries", str(metrics.n_queries))

    console.print(table)


def _print_generation_table(metrics) -> None:
    table = Table(title="Generation Metrics  (RAG via Ollama)", box=box.SIMPLE_HEAD)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_row("ROUGE-L F1", f"{metrics.rouge_l_f1:.4f}")
    table.add_row("Pairs evaluated", str(metrics.n_evaluated))
    console.print(table)


if __name__ == "__main__":
    main()
