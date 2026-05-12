"""
Text normalization for arXiv paper abstracts and titles.

Handles LaTeX artifacts, Unicode normalization, and whitespace cleanup.
"""

import re
import unicodedata
from dataclasses import replace

from src.corpus.downloader import Paper


# Common LaTeX command patterns to strip
_LATEX_CMD_RE = re.compile(
    r"\\(?:textbf|textit|emph|mathrm|mathbf|mathit|mathcal|text|cite|ref|label|footnote)"
    r"\{([^}]*)\}",
    re.IGNORECASE,
)
_LATEX_MATH_INLINE_RE = re.compile(r"\$[^$]{0,80}\$")
_LATEX_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_BACKSLASH_CMD_RE = re.compile(r"\\[a-zA-Z]+\*?(?:\{[^}]*\}|\[[^\]]*\])*")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_CITATION_RE = re.compile(r"\[[\d,\s]+\]")


def normalize_text(text: str) -> str:
    """
    Clean a single text string for embedding quality.

    Steps:
    1. Unicode NFC normalization
    2. Strip HTML tags
    3. Replace inline math with <MATH> placeholder
    4. Unwrap common LaTeX commands, keeping inner text
    5. Strip remaining LaTeX backslash commands
    6. Remove numeric citation brackets like [1,2]
    7. Collapse whitespace
    """
    # NFC normalization — handles accented chars, ligatures
    text = unicodedata.normalize("NFC", text)

    # Remove HTML tags (sometimes present in arXiv HTML exports)
    text = _HTML_TAG_RE.sub(" ", text)

    # Display math first (multi-line), then inline
    text = _LATEX_DISPLAY_MATH_RE.sub(" <MATH> ", text)
    text = _LATEX_MATH_INLINE_RE.sub(" <MATH> ", text)

    # Unwrap e.g. \textbf{word} → word
    text = _LATEX_CMD_RE.sub(r"\1", text)

    # Strip remaining LaTeX commands
    text = _LATEX_BACKSLASH_CMD_RE.sub(" ", text)

    # Remove citation brackets
    text = _CITATION_RE.sub("", text)

    # Collapse horizontal whitespace (preserve single newlines for paragraph detection)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    return text.strip()


def normalize_paper(paper: Paper) -> Paper:
    """Return a copy of the paper with normalized title and abstract."""
    return replace(
        paper,
        title=normalize_text(paper.title),
        abstract=normalize_text(paper.abstract),
    )


def word_count(text: str) -> int:
    """Count whitespace-delimited tokens."""
    return len(text.split())
