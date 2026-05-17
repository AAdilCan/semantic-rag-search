"""Tests for text normalization."""

import pytest

from src.preprocessing.normalize import normalize_text, word_count


def test_remove_latex_inline_math():
    text = "The model $f(x)$ maps inputs to outputs."
    result = normalize_text(text)
    assert "$" not in result
    assert "<MATH>" in result


def test_remove_html_tags():
    text = "This is <b>important</b> text."
    result = normalize_text(text)
    assert "<b>" not in result
    assert "important" in result


def test_preserve_core_content():
    text = "Deep learning achieves state-of-the-art results on benchmarks."
    result = normalize_text(text)
    assert "Deep learning" in result
    assert "state-of-the-art" in result


def test_collapse_horizontal_whitespace():
    text = "word1   word2\t\tword3"
    result = normalize_text(text)
    assert "  " not in result


def test_latex_command_unwrap():
    text = r"The \textbf{transformer} model uses \textit{attention}."
    result = normalize_text(text)
    assert "transformer" in result
    assert "attention" in result
    assert "\\textbf" not in result


def test_word_count_basic():
    assert word_count("one two three") == 3


def test_word_count_empty():
    assert word_count("") == 0


def test_word_count_extra_spaces():
    assert word_count("  spaces  everywhere  ") == 2


def test_idempotent():
    text = "A standard sentence with no special chars."
    assert normalize_text(normalize_text(text)) == normalize_text(text)


def test_citation_bracket_removal():
    text = "This result was shown [1,2,3] in prior work [4]."
    result = normalize_text(text)
    assert "[1,2,3]" not in result
    assert "prior work" in result
