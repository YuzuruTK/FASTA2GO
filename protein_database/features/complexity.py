"""Sequence complexity analysis for protein sequences."""

import math
from dataclasses import dataclass

from protein_database.features.sequence_features import STANDARD_AMINO_ACIDS


@dataclass
class ComplexityResult:
    """Sequence complexity metrics for a protein sequence."""

    shannon_entropy: float
    amino_acid_diversity: float
    repeated_amino_acid_count: int
    low_complexity_region_count: int
    max_homopolymer_length: int


def shannon_entropy(sequence: str) -> float:
    """Compute the Shannon entropy of a protein sequence.

    Shannon entropy measures the information content / randomness of the
    sequence. Higher entropy = more diverse sequence. Maximum is log2(20)
    for the 20 standard amino acids.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Shannon entropy in bits (log2 scale).
    """
    if not sequence:
        return 0.0

    length = len(sequence)
    counts: dict[str, int] = {}
    for c in sequence:
        counts[c] = counts.get(c, 0) + 1

    entropy = 0.0
    for count in counts.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def amino_acid_diversity(sequence: str) -> float:
    """Compute amino acid diversity as the fraction of the 20 standard AAs present.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Diversity score between 0.0 and 1.0.
    """
    if not sequence:
        return 0.0

    unique = {c for c in sequence if c in STANDARD_AMINO_ACIDS}
    return len(unique) / len(STANDARD_AMINO_ACIDS)


def count_repeated_amino_acids(sequence: str, repeat_count: int = 2) -> int:
    """Count amino acids that appear at least `repeat_count` times.

    Args:
        sequence: Amino acid sequence.
        repeat_count: Minimum number of occurrences to count as repeated.

    Returns:
        Number of distinct amino acids meeting the repeat threshold.
    """
    counts: dict[str, int] = {}
    for c in sequence:
        counts[c] = counts.get(c, 0) + 1
    return sum(1 for count in counts.values() if count >= repeat_count)


def find_low_complexity_regions(
    sequence: str, min_run_length: int = 4
) -> int:
    """Detect low-complexity regions (homopolymer runs) in a sequence.

    A low-complexity region is a consecutive run of the same amino acid
    (homopolymer) with length >= min_run_length.

    Args:
        sequence: Amino acid sequence.
        min_run_length: Minimum run length to count as low-complexity.

    Returns:
        Number of low-complexity regions found.
    """
    if len(sequence) < min_run_length:
        return 0

    count = 0
    run_length = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            run_length += 1
        else:
            if run_length >= min_run_length:
                count += 1
            run_length = 1

    # Check the last run
    if run_length >= min_run_length:
        count += 1

    return count


def max_homopolymer_length(sequence: str) -> int:
    """Find the longest consecutive run of a single amino acid.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Length of the longest homopolymer run.
    """
    if not sequence:
        return 0

    max_run = 1
    current_run = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    return max_run


def compute_complexity(sequence: str) -> ComplexityResult:
    """Compute all sequence complexity metrics.

    Args:
        sequence: Amino acid sequence.

    Returns:
        ComplexityResult with all computed values.
    """
    return ComplexityResult(
        shannon_entropy=shannon_entropy(sequence),
        amino_acid_diversity=amino_acid_diversity(sequence),
        repeated_amino_acid_count=count_repeated_amino_acids(sequence),
        low_complexity_region_count=find_low_complexity_regions(sequence),
        max_homopolymer_length=max_homopolymer_length(sequence),
    )