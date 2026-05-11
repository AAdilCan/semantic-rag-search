"""
Corpus exploration — loads downloaded papers and produces summary statistics
and visualizations saved to figures/.

Usage:
    python scripts/explore_corpus.py --input data/raw/cs_CL_papers.jsonl
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        rows.append(
            {
                "paper_id": r["paper_id"],
                "title": r["title"],
                "abstract": r["abstract"],
                "n_authors": len(r["authors"]),
                "n_categories": len(r["categories"]),
                "published": r["published"],
                "abstract_words": len(r["abstract"].split()),
                "title_words": len(r["title"].split()),
            }
        )
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"])
    return df


def plot_abstract_length_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["abstract_words"], bins=50, kde=True, ax=ax)
    ax.set_xlabel("Abstract length (words)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of abstract lengths")
    fig.tight_layout()
    fig.savefig(out_dir / "abstract_length_dist.png", dpi=150)
    plt.close(fig)
    logger.info("Saved abstract length distribution")


def plot_papers_over_time(df: pd.DataFrame, out_dir: Path) -> None:
    monthly = df.groupby(df["published"].dt.to_period("M")).size().reset_index()
    monthly.columns = ["month", "count"]
    monthly["month_str"] = monthly["month"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(monthly["month_str"], monthly["count"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Papers published")
    ax.set_title("Papers per month")
    # Show every 3rd label to avoid crowding
    xticks = ax.get_xticklabels()
    for i, t in enumerate(xticks):
        if i % 3 != 0:
            t.set_visible(False)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "papers_over_time.png", dpi=150)
    plt.close(fig)
    logger.info("Saved papers over time plot")


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Corpus Summary ===")
    print(f"  Total papers:         {len(df)}")
    print(f"  Date range:           {df['published'].min().date()} → {df['published'].max().date()}")
    print(f"  Abstract length:      mean={df['abstract_words'].mean():.1f}  "
          f"median={df['abstract_words'].median():.0f}  "
          f"std={df['abstract_words'].std():.1f}")
    print(f"  Title length:         mean={df['title_words'].mean():.1f}")
    print(f"  Avg authors/paper:    {df['n_authors'].mean():.1f}")
    total_words = df["abstract_words"].sum()
    print(f"  Total words (abstracts): {total_words:,}")
    print()


def main(input_path: str) -> None:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    FIGURES_DIR.mkdir(exist_ok=True)

    records = load_jsonl(path)
    logger.info("Loaded %d records from %s", len(records), path)

    df = build_dataframe(records)
    print_summary(df)

    plot_abstract_length_distribution(df, FIGURES_DIR)
    plot_papers_over_time(df, FIGURES_DIR)

    # Word frequency in titles
    all_title_words: list[str] = []
    for title in df["title"]:
        all_title_words.extend(w.lower() for w in title.split() if len(w) > 3)
    freq = Counter(all_title_words).most_common(20)
    print("Top 20 title words:")
    for word, count in freq:
        print(f"  {word:<20} {count}")

    print(f"\nFigures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore downloaded arXiv corpus.")
    parser.add_argument("--input", required=True, help="Path to JSONL corpus file.")
    args = parser.parse_args()
    main(args.input)
