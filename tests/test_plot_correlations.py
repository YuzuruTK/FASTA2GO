"""Tests for loading and rendering the correlation matrices."""

from pathlib import Path

import numpy as np

from scripts.plot_correlations import (
    is_symmetric,
    load_matrix,
    plot_matrix,
    scale_for,
)


def _write_matrix(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_matrix_parses_labels_and_values(tmp_path: Path) -> None:
    path = _write_matrix(
        tmp_path / "sample.csv",
        "term,a,b\n"
        "a,1.0,0.5\n"
        "b,0.5,1.0\n",
    )
    rows, columns, values = load_matrix(path)
    assert rows == ["a", "b"]
    assert columns == ["a", "b"]
    assert np.array_equal(values, np.array([[1.0, 0.5], [0.5, 1.0]]))


def test_is_symmetric_true_for_correlation_matrix() -> None:
    values = np.array([[1.0, -0.2], [-0.2, 1.0]])
    assert is_symmetric(values) is True


def test_is_symmetric_false_for_rectangular_or_asymmetric() -> None:
    assert is_symmetric(np.zeros((2, 3))) is False
    assert is_symmetric(np.array([[1.0, 0.2], [0.1, 1.0]])) is False


def test_scale_for_signed_correlation_is_centred_and_symmetric() -> None:
    values = np.array([[1.0, -0.4], [-0.4, 1.0]])
    style = scale_for("words_phi", values)
    assert style["center"] == 0.0
    assert style["vmin"] == -style["vmax"]


def test_scale_for_jaccard_is_zero_to_max() -> None:
    values = np.array([[1.0, 0.3], [0.3, 1.0]])
    style = scale_for("words_jaccard", values)
    assert style["vmin"] == 0.0
    assert style["vmax"] == 1.0


def test_plot_matrix_writes_png(tmp_path: Path) -> None:
    path = _write_matrix(
        tmp_path / "words_phi.csv",
        "term,a,b,c\n"
        "a,1.0,0.5,-0.2\n"
        "b,0.5,1.0,0.1\n"
        "c,-0.2,0.1,1.0\n",
    )
    out = plot_matrix(path, tmp_path / "plots", dpi=72)
    assert out.exists()
    assert out.stat().st_size > 0
