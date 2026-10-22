"""Pure helpers for building per-protein keyword terms.

Kept import-light (no DuckDB) so the logic is trivially testable.
"""

from __future__ import annotations

import re

#: One GO cell segment: "ATP binding [GO:0005524]".
#: The term cannot contain '[', so matching is linear (no backtracking).
GO_CELL = re.compile(r"^(?P<term>[^[]+)\[(?P<id>GO:\d+)\]$")


def parse_go_cell(cell: str | None) -> list[tuple[str, str]]:
    """Parse a UniProt GO cell into (term, GO id) pairs.

    A cell holds ``;``-separated ``term [GO:xxxxxxx]`` segments. Empty cells
    and non-matching segments produce nothing rather than raising.
    """
    if not cell:
        return []
    pairs: list[tuple[str, str]] = []
    for segment in cell.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        match = GO_CELL.match(segment)
        if match:
            pairs.append((match.group("term").strip(), match.group("id")))
    return pairs


def phrase_key(tokens: list[str] | tuple[str, ...]) -> str:
    """Join already-clean tokens into the stored phrase form ("plasma_membrane")."""
    return "_".join(tokens)


def build_phrase_index(
    tokenized_terms: list[tuple[str, ...]],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """Index multi-token phrases by their first token.

    Only phrases with at least two tokens are indexed: a single-token GO term
    is a word, not a phrase, and already lands in the ``words`` column.
    Candidate order is deterministic longest-first, so the most specific
    concept comes first when inspecting matches.
    """
    index: dict[str, set[tuple[str, ...]]] = {}
    for tokens in tokenized_terms:
        if len(tokens) < 2:
            continue
        index.setdefault(tokens[0], set()).add(tokens)
    return {
        head: tuple(sorted(candidates, key=lambda c: (-len(c), c)))
        for head, candidates in index.items()
    }


def find_phrases(
    tokens: list[str],
    index: dict[str, tuple[tuple[str, ...], ...]],
) -> set[str]:
    """Return every indexed phrase occurring contiguously in ``tokens``.

    Overlapping phrases are all reported (both ``atp_binding`` and
    ``atp_binding_cassette`` when present) because each is a meaningful
    grouping term on its own.
    """
    found: set[str] = set()
    for i, token in enumerate(tokens):
        for candidate in index.get(token, ()):
            if tuple(tokens[i : i + len(candidate)]) == candidate:
                found.add(phrase_key(candidate))
    return found



def keep_term(df: int, n: int, min_occ: float = 0.05, max_occ: float = 0.95) -> bool:
    """Return True when a term's document frequency is within [min_occ, max_occ].

    Both ends are inclusive: df/N exactly equal to a bound is kept.
    """
    if n <= 0:
        return False
    occ = df / n
    return min_occ <= occ <= max_occ
