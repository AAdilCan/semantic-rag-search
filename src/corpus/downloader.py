"""
Fetch paper abstracts from arXiv and cache them as JSONL.

Usage:
    python -m src.corpus.downloader --category cs.CL --max-papers 1000
    python -m src.corpus.downloader --category cs.LG --max-papers 500 --output data/raw/lg_papers.jsonl
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import arxiv
import click
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published: str
    url: str


def fetch_papers(
    category: str,
    max_papers: int,
    date_start: str | None = None,
    date_end: str | None = None,
) -> list[Paper]:
    """
    Fetch up to max_papers arXiv abstracts for a given category.

    Args:
        category: arXiv category string, e.g. 'cs.CL', 'cs.LG'.
        max_papers: Maximum number of papers to retrieve.
        date_start: ISO date string 'YYYY-MM-DD' for earliest submission (optional).
        date_end: ISO date string 'YYYY-MM-DD' for latest submission (optional).

    Returns:
        List of Paper dataclass instances.
    """
    query = f"cat:{category}"
    if date_start and date_end:
        # arXiv date filter uses submittedDate field
        ds = date_start.replace("-", "") + "000000"
        de = date_end.replace("-", "") + "235959"
        query += f" AND submittedDate:[{ds} TO {de}]"

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    for result in tqdm(client.results(search), total=max_papers, desc=f"Fetching {category}"):
        paper = Paper(
            paper_id=result.entry_id.split("/")[-1],
            title=result.title.strip(),
            abstract=result.summary.strip(),
            authors=[a.name for a in result.authors],
            categories=result.categories,
            published=result.published.date().isoformat(),
            url=result.entry_id,
        )
        papers.append(paper)
        if len(papers) >= max_papers:
            break

    return papers


def save_jsonl(papers: list[Paper], output_path: Path) -> None:
    """Write papers to a JSONL file, one JSON object per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
    logger.info("Saved %d papers to %s", len(papers), output_path)


def load_jsonl(path: Path) -> list[Paper]:
    """Load papers from a JSONL file."""
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                papers.append(Paper(**data))
    return papers


def corpus_stats(papers: list[Paper]) -> dict:
    """Compute basic corpus statistics."""
    abstract_lengths = [len(p.abstract.split()) for p in papers]
    title_lengths = [len(p.title.split()) for p in papers]
    all_cats: list[str] = []
    for p in papers:
        all_cats.extend(p.categories)

    from collections import Counter

    cat_counts = Counter(all_cats)
    date_range = (
        min(p.published for p in papers),
        max(p.published for p in papers),
    )

    return {
        "total_papers": len(papers),
        "avg_abstract_words": round(sum(abstract_lengths) / len(abstract_lengths), 1),
        "min_abstract_words": min(abstract_lengths),
        "max_abstract_words": max(abstract_lengths),
        "avg_title_words": round(sum(title_lengths) / len(title_lengths), 1),
        "top_categories": cat_counts.most_common(10),
        "date_range": date_range,
    }


@click.command()
@click.option("--category", default="cs.CL", show_default=True, help="arXiv category to fetch.")
@click.option("--max-papers", default=1000, show_default=True, help="Maximum papers to download.")
@click.option(
    "--output",
    default=None,
    help="Output JSONL path. Defaults to data/raw/<category>_papers.jsonl.",
)
@click.option("--date-start", default=None, help="Filter papers from date YYYY-MM-DD.")
@click.option("--date-end", default=None, help="Filter papers up to date YYYY-MM-DD.")
def main(
    category: str,
    max_papers: int,
    output: str | None,
    date_start: str | None,
    date_end: str | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_path = Path(output) if output else Path(f"data/raw/{category.replace('.', '_')}_papers.jsonl")

    if out_path.exists():
        existing = load_jsonl(out_path)
        logger.info("Cache hit: %d papers already at %s", len(existing), out_path)
        stats = corpus_stats(existing)
        _print_stats(stats)
        return

    logger.info("Fetching up to %d papers from arXiv category %s", max_papers, category)
    papers = fetch_papers(category, max_papers, date_start, date_end)

    if not papers:
        logger.error("No papers fetched — check category or network.")
        return

    save_jsonl(papers, out_path)
    stats = corpus_stats(papers)
    _print_stats(stats)


def _print_stats(stats: dict) -> None:
    print("\n--- Corpus Statistics ---")
    print(f"  Papers:           {stats['total_papers']}")
    print(f"  Abstract words:   avg={stats['avg_abstract_words']}  "
          f"min={stats['min_abstract_words']}  max={stats['max_abstract_words']}")
    print(f"  Title words:      avg={stats['avg_title_words']}")
    print(f"  Date range:       {stats['date_range'][0]}  →  {stats['date_range'][1]}")
    print(f"  Top categories:   {stats['top_categories'][:5]}")
    print()


if __name__ == "__main__":
    main()
