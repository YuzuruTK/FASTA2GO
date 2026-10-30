"""FASTA2GO extraction stage (FASTA -> DuckDB).

``go_slim`` holds the 41-label Molecular Function vocabulary and its propagation
rules; it is used by ``etapa1/build_go_slim_labels.py`` and by the exploratory
analysis, and it never writes to the database.
"""

__all__ = ["db", "fasta", "features", "go_slim", "pipeline", "uniprot"]
