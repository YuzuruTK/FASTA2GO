"""SQLAlchemy ORM models for the protein feature database."""

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Protein(Base):
    """Stores original FASTA information for each protein."""

    __tablename__ = "proteins"
    __table_args__ = (
        UniqueConstraint("fasta_id", "source_file", name="uq_protein_fasta_source"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fasta_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    sequence: Mapped[str] = mapped_column(Text, nullable=False)
    sequence_length: Mapped[int] = mapped_column(Integer, nullable=False)
    source_file: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    sequence_features: Mapped["SequenceFeature"] = relationship(
        back_populates="protein", uselist=False, cascade="all, delete-orphan"
    )
    amino_acid_composition: Mapped["AminoAcidComposition"] = relationship(
        back_populates="protein", uselist=False, cascade="all, delete-orphan"
    )
    validation_features: Mapped["ValidationFeature"] = relationship(
        back_populates="protein", uselist=False, cascade="all, delete-orphan"
    )


class SequenceFeature(Base):
    """Stores calculated physicochemical properties of a protein sequence."""

    __tablename__ = "sequence_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    protein_id: Mapped[int] = mapped_column(
        ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False, index=True
    )
    molecular_weight: Mapped[float] = mapped_column(Float, nullable=True)
    isoelectric_point: Mapped[float] = mapped_column(Float, nullable=True)
    gravy_index: Mapped[float] = mapped_column(Float, nullable=True)
    instability_index: Mapped[float] = mapped_column(Float, nullable=True)
    aliphatic_index: Mapped[float] = mapped_column(Float, nullable=True)
    aromaticity: Mapped[float] = mapped_column(Float, nullable=True)
    charge_positive_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    charge_negative_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    hydrophobic_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    polar_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    charged_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    sulfur_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    aromatic_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    sequence_entropy: Mapped[float] = mapped_column(Float, nullable=True)
    amino_acid_diversity: Mapped[float] = mapped_column(Float, nullable=True)

    protein: Mapped["Protein"] = relationship(back_populates="sequence_features")


class AminoAcidComposition(Base):
    """Stores amino acid frequency percentages for a protein sequence."""

    __tablename__ = "amino_acid_composition"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    protein_id: Mapped[int] = mapped_column(
        ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False, index=True
    )

    alanine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    arginine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    asparagine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    aspartic_acid_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    cysteine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    glutamine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    glutamic_acid_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    glycine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    histidine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    isoleucine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    leucine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    lysine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    methionine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    phenylalanine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    proline_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    serine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    threonine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    tryptophan_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    tyrosine_percentage: Mapped[float] = mapped_column(Float, nullable=True)
    valine_percentage: Mapped[float] = mapped_column(Float, nullable=True)

    protein: Mapped["Protein"] = relationship(back_populates="amino_acid_composition")


class ValidationFeature(Base):
    """Stores future validation metrics. Fields initially NULL."""

    __tablename__ = "validation_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    protein_id: Mapped[int] = mapped_column(
        ForeignKey("proteins.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence_quality_score: Mapped[float] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float] = mapped_column(Float, nullable=True)
    solubility_score: Mapped[float] = mapped_column(Float, nullable=True)
    structure_score: Mapped[float] = mapped_column(Float, nullable=True)
    overall_score: Mapped[float] = mapped_column(Float, nullable=True)
    classification: Mapped[str] = mapped_column(String(100), nullable=True)

    protein: Mapped["Protein"] = relationship(back_populates="validation_features")