#!/usr/bin/env python3
"""Build the curated functional-keyword CSV for protein-function classification.

Reads the top-N word-frequency vocabulary (``data/word_frequencies.csv``,
produced by ``scripts/word_count_test.py --export-top``) and emits:

* ``data/functional_labels.csv`` with the schema::

      word,family,class,weight,strength,reason

* ``data/functional_labels_discarded.csv`` with the schema::

      word,reason

Design goals (precision over quantity):

* A word is evidence for one or more broad classes; words may belong to
  multiple classes with different weights.
* Morphological variants are grouped into conceptual *families*
  (catalysis, transport, regulation, activation, ...) using biological
  meaning — no generic stemming.
* Generic/ambiguous words (ATP, binding, membrane, ...) receive low
  weights and are never treated as definitive classifiers.
* Words that provide no functional discrimination (grammatical words,
  amino-acid names, organism descriptors, positional words, ...) are
  discarded to a separate file.
* Only words present in the input vocabulary are emitted.
* Deterministic and reproducible: edit the rules below and re-run;
  never hand-edit the generated CSVs.

Usage:
    uv run python scripts/build_functional_labels.py
    uv run python scripts/build_functional_labels.py \
        --input data/word_frequencies.csv --output-dir data
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "word_frequencies.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data"

#: Broad functional classes, in canonical order for reporting.
CLASSES = [
    "enzyme",
    "transporter",
    "dna_rna_binding",
    "structural",
    "receptor_signaling",
    "motor",
    "defense",
    "storage",
]

# ---------------------------------------------------------------------------
# Weight -> strength mapping
# ---------------------------------------------------------------------------


def strength_for(weight: float) -> str:
    """Map a numeric weight to its qualitative strength label."""
    if weight >= 0.8:
        return "strong"
    if weight >= 0.5:
        return "medium"
    return "weak"


# ---------------------------------------------------------------------------
# Curated functional keyword rules.
#
# word -> list of (family, class, weight, reason)
# strength is derived from weight. A word may carry multiple entries
# (multiple classes and/or families).
#
# Every word here was verified against the actual top-1000 vocabulary of
# the corpus; absent words are silently skipped at generation time.
# ---------------------------------------------------------------------------

KEYWORD_RULES: dict[str, list[tuple[str, str, float, str]]] = {}

def rule(family: str, cls: str, weight: float, reason: str, *words: str) -> None:
    """Register one or more words under a family/class/weight entry."""
    for word in words:
        KEYWORD_RULES.setdefault(word, []).append((family, cls, weight, reason))


# ===========================================================================
# ENZYME
# ===========================================================================

rule("catalysis", "enzyme", 1.0,
     "Directly indicates enzymatic catalysis",
     "catalyzes", "catalyze", "catalyzed", "catalyzing", "catalytic",
     "catalysis")
rule("catalysis", "enzyme", 0.9,
     "Names an enzyme or a specific catalytic reaction step",
     "enzyme", "enzymes", "reaction", "reactions", "hydrolysis",
     "hydrolyzes")
rule("proteolysis", "enzyme", 0.9,
     "Indicates peptide-bond cleavage by a proteolytic enzyme",
     "cleaves", "cleave", "cleavage", "degrades")
rule("ec_class", "enzyme", 1.0,
     "Names an EC enzyme family directly",
     "kinase", "synthase", "reductase", "oxidoreductase", "dehydrogenase",
     "transferase", "ligase", "hydrolase", "isomerase", "oxidase",
     "monooxygenase", "phosphatase", "protease", "synthetase")
rule("ec_class", "enzyme", 0.9,
     "Names a specific enzyme or enzyme complex",
     "atpase", "atpases", "gtpase", "cyclase", "aldol", "exonuclease",
     "p450", "ndh", "bifunctional")
rule("reaction_chemistry", "enzyme", 0.7,
     "Names a specific biochemical reaction catalyzed by enzymes",
     "oxidation", "isomerization", "transamidation", "deamination",
     "carboxylation", "hydroxylation", "decarboxylation",
     "phosphorylates", "methylates")
rule("reaction_chemistry", "enzyme", 0.6,
     "Describes a chemical transformation typical of enzyme activity",
     "phosphorylation", "reduction", "conversion", "converts",
     "converting", "converted", "convert")
rule("reaction_chemistry", "enzyme", 0.4,
     "Weakly indicates an enzymatic transformation in context",
     "degraded", "degradation", "acetyl", "methyltransferase")
rule("group_transfer", "enzyme", 0.3,
     "Generic transfer wording; only weak evidence of enzymatic activity",
     "transfer", "transfers", "transferred")
rule("redox", "enzyme", 0.5,
     "Redox chemistry implies oxidoreductase activity",
     "redox", "oxidative")
rule("redox", "enzyme", 0.3,
     "Electron transfer wording associated with redox enzymes",
     "electron", "electrons")
rule("ubiquitination", "enzyme", 0.4,
     "Ubiquitin conjugation implies ligase-associated activity",
     "ubiquitin", "ubiquitination", "e3")
rule("proteasome", "enzyme", 0.5,
     "Proteasomal degradation is proteolytic enzyme activity",
     "proteasome", "proteasomal", "proteolytic")
rule("aminoacyl", "enzyme", 0.6,
     "Aminoacylation indicates tRNA-synthetase activity",
     "aminoacyl")
rule("methylation", "enzyme", 0.5,
     "Methylation indicates methyltransferase activity",
     "methylation")

# --- cofactors (weak-to-moderate enzyme evidence) --------------------------

rule("cofactor_redox", "enzyme", 0.4,
     "Redox cofactor strongly associated with oxidoreductases",
     "nad", "nadh", "nadph", "fad", "fmn")
rule("cofactor_redox", "enzyme", 0.3,
     "Quinone electron carrier associated with redox enzymes",
     "quinones", "plastoquinone", "ubiquinone", "ubiquinol")
rule("cofactor", "enzyme", 0.3,
     "Enzyme cofactor; weak evidence without further context",
     "coa", "acp", "pyridoxal", "thiamine", "biotin", "cofactor",
     "coenzyme")
rule("nucleotide_energy", "enzyme", 0.3,
     "Energy nucleotide; ambiguous between enzyme and motor",
     "atp", "adp", "gtp")
rule("nucleotide", "enzyme", 0.2,
     "Nucleotide moiety; very weak evidence of enzymatic activity",
     "gdp", "udp", "ctp", "gmp", "ump", "phosphate")
rule("metal_cofactor", "enzyme", 0.2,
     "Metal ion often used as enzymatic cofactor",
     "iron", "fe", "zinc", "copper")
rule("metal_cofactor", "dna_rna_binding", 0.2,
     "Zinc is characteristic of zinc-finger DNA/RNA-binding motifs",
     "zinc")
rule("heme", "enzyme", 0.2,
     "Heme proteins are frequently enzymes but not exclusively",
     "heme")

# ===========================================================================
# GENERIC BINDING (ambiguous; low weight everywhere)
# ===========================================================================

rule("general_binding", "enzyme", 0.2,
     "Generic binding is not sufficient to determine function",
     "binding", "binds", "bind", "bound")
rule("general_binding", "transporter", 0.2,
     "Generic binding is not sufficient to determine function",
     "binding", "binds", "bind", "bound")
rule("general_binding", "receptor_signaling", 0.2,
     "Generic binding is not sufficient to determine function",
     "binding", "binds", "bind", "bound")
rule("general_binding", "dna_rna_binding", 0.2,
     "Generic binding is not sufficient to determine function",
     "binding", "binds", "bind", "bound")

# ===========================================================================
# TRANSPORTER
# ===========================================================================

rule("transport", "transporter", 1.0,
     "Directly indicates transport function",
     "transporter", "transport", "transports")
rule("transport", "transporter", 0.9,
     "Indicates movement of molecules across membranes or compartments",
     "translocation", "channel", "channels", "uptake", "import", "efflux",
     "translocase")
rule("transport", "transporter", 0.8,
     "Strong transport-related evidence",
     "translocated", "pump", "shuttles")
rule("transport", "transporter", 0.7,
     "Membrane translocation or trafficking machinery",
     "transmembrane", "carrier", "secretion", "trafficking")
rule("transport", "transporter", 0.6,
     "Ion-gradient transport context",
     "gradient", "proton", "protons", "sodium", "potassium")
rule("transport", "transporter", 0.5,
     "Ion transport context",
     "ion", "ions")
rule("transport", "transporter", 0.3,
     "Generic membrane association, not necessarily transport",
     "membrane", "membranes")
rule("transport_system", "transporter", 0.9,
     "Names a known transporter family",
     "abc")
rule("transport_system", "transporter", 0.6,
     "Sec pathway mediates protein export across membranes",
     "sec")
rule("vesicular", "transporter", 0.6,
     "Vesicle-mediated transport process",
     "endocytosis")
rule("vesicular", "transporter", 0.3,
     "Weak vesicular-trafficking association",
     "fusion", "recycling", "gated", "voltage")
rule("protein_targeting", "transporter", 0.2,
     "Targeting wording suggests intracellular traffic; very weak",
     "targeting")

# ===========================================================================
# DNA_RNA_BINDING
# ===========================================================================

rule("nucleic_acid", "dna_rna_binding", 1.0,
     "Directly indicates nucleic-acid association",
     "dna", "rna", "rrna", "trna", "trnas", "dsdna")
rule("nucleic_acid", "dna_rna_binding", 0.9,
     "Names an RNA species or RNA-processing activity",
     "mrna", "mrnas", "rnas", "polymerase", "helicase")
rule("ribosomal", "dna_rna_binding", 1.0,
     "Directly indicates ribosome association",
     "ribosomal", "ribosome", "ribosomes")
rule("rrna_subunit", "dna_rna_binding", 0.8,
     "rRNA component designation",
     "16s", "23s")
rule("ribosome_subunit", "dna_rna_binding", 0.7,
     "Ribosomal subunit designation",
     "30s", "50s", "70s", "5s")
rule("central_dogma", "dna_rna_binding", 0.9,
     "Directly indicates transcription or translation activity",
     "transcription", "transcriptional", "translation", "translational",
     "replication", "recombination")
rule("central_dogma", "dna_rna_binding", 0.7,
     "Genome maintenance or repair activity",
     "repair", "genome", "genomic")
rule("chromatin", "dna_rna_binding", 0.9,
     "Directly indicates chromatin association",
     "chromatin", "histone", "histones")
rule("chromatin", "dna_rna_binding", 0.8,
     "Chromosome-level association",
     "chromosome", "chromosomal")
rule("regulation", "dna_rna_binding", 0.9,
     "Promoter region association",
     "promoter", "promoters")
rule("regulation", "dna_rna_binding", 0.7,
     "Transcriptional repression activity",
     "repressor", "repression", "represses")
rule("nucleic_acid", "dna_rna_binding", 0.8,
     "Nucleic-acid enzymology or structure",
     "endonuclease", "exonuclease", "operon", "operons", "codons",
     "splicing", "anticodon", "ribonucleoprotein")
rule("nucleic_acid", "dna_rna_binding", 0.6,
     "Nucleic-acid related term; moderate evidence",
     "nucleic", "template", "strand", "stranded", "editing",
     "ribonucleotide", "elongation", "capping", "misacylated",
     "mischarged", "ruva", "ruvb", "ruvc", "holliday")
rule("nucleic_acid", "dna_rna_binding", 0.5,
     "Weak nucleic-acid association in context",
     "nucleotide", "nucleotides", "fork", "forks", "lesions", "uracil")
rule("nucleic_acid", "dna_rna_binding", 0.4,
     "Nucleoside/nucleobase metabolism context; weak evidence",
     "nucleoside", "ribonucleoside", "uridine", "adenosine", "guanine",
     "cytidine", "adenine", "guanosine")

# ===========================================================================
# STRUCTURAL
# ===========================================================================

rule("structure", "structural", 1.0,
     "Directly indicates a structural role",
     "structural")
rule("cytoskeleton", "structural", 0.9,
     "Names a cytoskeletal component",
     "scaffold", "cytoskeleton", "actin", "microtubule", "microtubules")
rule("filament", "structural", 0.8,
     "Filamentous structural assembly",
     "filaments", "filament")
rule("structure", "structural", 0.7,
     "Structural substructure",
     "stalk", "capsid", "spindle")
rule("structure", "structural", 0.6,
     "Generic structural wording; moderate evidence",
     "movement", "structure")
rule("structure", "structural", 0.5,
     "Weak structural evidence; common in generic prose",
     "structures", "backbone", "helix", "helices")

# ===========================================================================
# RECEPTOR_SIGNALING
# ===========================================================================

rule("receptor", "receptor_signaling", 1.0,
     "Directly indicates receptor function",
     "receptor", "receptors")
rule("signaling", "receptor_signaling", 0.9,
     "Directly indicates signal-transduction activity",
     "signaling", "signal", "signals", "transduction", "cascade")
rule("signaling", "receptor_signaling", 0.8,
     "Regulatory or sensory signaling role",
     "sensor", "regulator", "regulatory")
rule("regulation", "receptor_signaling", 0.8,
     "Indicates regulation of a biological process",
     "regulates", "regulation", "regulating", "regulated")
rule("activation", "receptor_signaling", 0.7,
     "Indicates activation in a signaling context",
     "activation", "activated", "activates", "activating", "activator",
     "effector")
rule("response", "receptor_signaling", 0.6,
     "Cellular response wording; moderate signaling evidence",
     "response", "responses")
rule("modulation", "receptor_signaling", 0.7,
     "Indicates modulation of a signaling process",
     "modulation", "modulates", "modulating", "modulate")
rule("stimulation", "receptor_signaling", 0.7,
     "Stimulus-response wording",
     "stimulation", "stimulated", "stimulates")
rule("induction", "receptor_signaling", 0.5,
     "Induction wording; moderate signaling evidence",
     "induces", "induced", "inducing", "induce")
rule("ligand", "receptor_signaling", 0.4,
     "Ligand wording; needs receptor context to be definitive",
     "ligand")
rule("hormone", "receptor_signaling", 0.7,
     "Hormone-related signaling",
     "hormone", "insulin")
rule("immune_signaling", "receptor_signaling", 0.6,
     "Cytokine-like immune signaling",
     "interferon")
rule("gtpase_switch", "receptor_signaling", 0.5,
     "GTPases act as signaling switches",
     "gtpase")
rule("phosphorylation", "receptor_signaling", 0.5,
     "Phosphorylation cascades are core signaling events",
     "phosphorylation")
rule("second_messenger", "receptor_signaling", 0.2,
     "Calcium acts as a second messenger; very weak alone",
     "calcium")

# ===========================================================================
# MOTOR
# ===========================================================================

rule("motor", "motor", 1.0,
     "Directly indicates motor activity",
     "motor", "motility")
rule("motor", "motor", 0.9,
     "Rotary motion generation",
     "rotary")
rule("motor", "motor", 0.6,
     "Movement wording; shared with structural class",
     "movement")

# ===========================================================================
# DEFENSE
# ===========================================================================

rule("defense", "defense", 1.0,
     "Directly indicates defense function",
     "defense", "antimicrobial")
rule("immunity", "defense", 0.9,
     "Directly indicates immune function",
     "immune", "immunity", "antiviral", "virulence")
rule("toxin", "defense", 0.9,
     "Toxin activity",
     "toxin")
rule("toxin", "defense", 0.7,
     "Toxicity wording",
     "toxic", "toxicity")
rule("antigen", "defense", 0.8,
     "Antigen-related immune function",
     "antigen")
rule("protection", "defense", 0.8,
     "Resistance to hostile conditions or agents",
     "resistance")
rule("infection", "defense", 0.7,
     "Infection context",
     "infection")
rule("infection", "defense", 0.5,
     "Infected-state context; weaker evidence",
     "infected")
rule("protection", "defense", 0.7,
     "Protective function wording",
     "protects", "protection")
rule("protection", "defense", 0.6,
     "Protective function wording (participial form)",
     "protecting")
rule("inflammation", "defense", 0.8,
     "Inflammatory response",
     "inflammatory", "innate")
rule("immune_signaling", "defense", 0.6,
     "Cytokine-like immune defense signaling",
     "interferon")
rule("effector", "defense", 0.5,
     "Effector proteins are often virulence factors",
     "effector")

# ===========================================================================
# DISCARD RULES
#
# word -> human-readable rejection reason. Applied only to words present
# in the input vocabulary.
# ===========================================================================

DISCARD_RULES: dict[str, str] = {}


def discard(reason: str, *words: str) -> None:
    """Register one or more words as discarded with a reason."""
    for word in words:
        DISCARD_RULES[word] = reason


discard(
    "Grammatical/common English word with no functional signal",
    "across", "either", "seems", "therefore", "however", "although",
    "whereas", "since", "without", "due", "toward", "around", "onto",
    "back", "less", "wide", "fully", "highly", "strongly", "weak",
    "rapid", "new", "final", "last", "three", "four", "following",
    "corresponding", "additional", "variety", "certain", "consist",
    "consists", "composed", "provided", "makes", "takes", "use", "uses",
    "occur", "occurs", "shows", "appears", "believed", "likely",
    "probably", "thought", "seen", "lack", "termed", "abstract",
    "prior",
    "subsequently", "initially", "initial", "premature", "close",
    "open", "main", "order", "place", "turn", "level", "levels",
    "range", "types", "ability", "quality", "physical", "properties",
    "selective", "preferentially", "preference", "indirectly",
    "negatively", "positively", "correctly", "accuracy", "efficiency",
    "optimized", "potent", "higher", "lower", "absence", "presence",
    "subset", "stage", "stages", "phase", "step", "steps", "event",
    "events", "state", "conditions", "environment", "space", "ends",
    "end", "terminus", "termini", "positions", "center", "adjacent",
    "internal", "apical", "outer", "inner", "peripheral", "integral",
    "upstream", "downstream", "cis", "beginning", "direction",
    "connecting", "contact", "contacts", "interface", "surface",
    "located", "localization", "targets", "target", "recognition",
    "recruits", "recruited", "recruitment", "association", "associates",
    "coupled", "coupling", "couples", "dependent", "mediated",
    "mediate", "linked", "free", "cross", "bridge", "bridges",
    "linking", "interacts", "interaction", "interactions", "isoform",
    "canonical", "cognate", "term", "wide", "dual", "secondary",
    "primary", "specificity", "active", "activities", "processing",
    "maturation", "precursor", "mature", "assembly", "organization",
)

discard(
    "Vague scientific prose verb; no class-discriminative signal",
    "promotes", "promoting", "promote", "controls", "control",
    "triggers", "enhances", "enhancing", "increases", "increase",
    "increasing", "generates", "generate", "generating", "produces",
    "produce", "producing", "produced", "yields", "yield", "yielding",
    "release", "releases", "leads", "leading", "results", "result",
    "causing", "causes", "prevents", "avoid", "blocking", "blocks",
    "removal", "removes", "requires", "displays", "exhibits",
    "possesses", "crucial", "implicated", "stabilizes", "stabilizing",
    "stabilize", "maintains", "maintain", "maintaining", "contributes",
    "contribute", "contributing", "facilitate", "facilitates",
    "facilitating", "accelerates", "accelerate", "directed",
    "directs", "directing", "targeted", "targets", "conserves",
    "conserve", "conserving", "confers", "confer", "drives", "drive",
    "driving", "captures", "capture", "resolves", "resolve",
    "introduces", "introduce", "finds", "find", "occurring",
)

discard(
    "Generic biological-context word describing cells or organisms, "
    "not protein function",
    "cells", "cellular", "cell", "host", "species", "organisms",
    "organism", "tissue", "tissues", "physiological", "biological",
    "growth", "development", "differentiation", "proliferation",
    "division", "embryonic", "neuronal", "neurons", "muscle", "brain",
    "blood", "macrophages", "epithelial", "endothelial", "synaptic",
    "nervous", "water", "light", "sugar", "nitrogen", "carbon",
    "oxygen", "hydrogen", "sulfur", "ammonia", "chloroplast",
    "mitochondria", "mitochondrial", "cytoplasm", "cytoplasmic",
    "cytosol", "cytosolic", "nucleus", "nuclear", "reticulum",
    "endoplasmic", "golgi", "plasma", "periplasmic", "extracellular",
    "intracellular", "envelope", "endosomal", "vesicles", "vesicle",
    "virion", "gram", "coli", "eukaryotic", "photosynthetic",
    "homeostasis", "morphogenesis", "survival", "death", "apoptosis",
    "autophagy", "stress", "heat", "shock", "metabolism", "pathway",
    "pathways", "biosynthesis", "synthesis", "acid", "acids",
    "amino", "peptide", "peptides", "polypeptide", "lipid", "lipids",
    "wall", "capsule", "matrix", "genetic", "expression", "genes",
    "gene", "genomes", "cellular_process",
)

discard(
    "Amino-acid name or code; does not indicate protein function",
    "alanine", "ala", "glycine", "serine", "ser", "lysine", "lys",
    "threonine", "thr", "methionine", "cysteine", "arginine",
    "proline", "pro", "valine", "isoleucine", "aspartate", "asp",
    "asparagine", "asn", "glutamate", "glu", "glutamine", "gln",
    "histidine", "tyrosine", "tyr", "seryl", "fmet", "formyl",
    "formylmethionyl", "cognate",
)

discard(
    "Experimental/contextual term from annotation prose",
    "vitro", "vivo", "concentration", "concentrations", "psi", "nm",
    "nano", "ph", "scan", "spontaneous", "spontaneously",
)

discard(
    "Organism/taxon descriptor rather than protein function",
    "bacterial", "viral", "virus", "human", "plant",
)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_vocabulary(path: Path) -> dict[str, int]:
    """Load the input vocabulary CSV (rank,word,proteins,total).

    Returns a mapping word -> rank (1-based). Raises SystemExit if the
    file is missing or malformed.
    """
    if not path.exists():
        print(f"Error: input vocabulary not found: {path}", file=sys.stderr)
        print(
            "Generate it first with:\n"
            "  uv run python scripts/word_count_test.py "
            "--exclude-file scripts/word_filter.txt "
            "--export-top 1000 --export-top-out data/word_frequencies.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    vocab: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"rank", "word"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            print(
                f"Error: {path} must have at least columns 'rank' and 'word'",
                file=sys.stderr,
            )
            sys.exit(1)
        for row in reader:
            word = row["word"].strip().lower()
            rank = int(row["rank"])
            if word:
                vocab[word] = rank
    return vocab


def build_label_rows(vocab: dict[str, int]) -> tuple[list[dict], set[str]]:
    """Match curated rules against the vocabulary.

    Returns (rows, matched_words) where rows are label records ready for
    CSV output and matched_words is the set of vocabulary words that
    received at least one label.
    """
    rows: list[dict] = []
    seen: set[tuple] = set()
    matched: set[str] = set()

    for word in sorted(vocab):
        # Isolated single letters are always noise.
        if len(word) == 1:
            continue
        for family, cls, weight, reason in KEYWORD_RULES.get(word, []):
            key = (word, family, cls)
            if key in seen:  # avoid duplicate rows
                continue
            seen.add(key)
            matched.add(word)
            rows.append(
                {
                    "word": word,
                    "family": family,
                    "class": cls,
                    "weight": f"{weight:.1f}",
                    "strength": strength_for(weight),
                    "reason": reason,
                }
            )
    return rows, matched


def build_discard_rows(
    vocab: dict[str, int], labeled: set[str]
) -> list[dict]:
    """Match discard rules against unlabeled vocabulary words."""
    rows: list[dict] = []
    for word in sorted(vocab):
        if len(word) == 1:
            rows.append({"word": word, "reason": "Isolated letter"})
            continue
        if word in labeled:
            continue
        reason = DISCARD_RULES.get(word)
        if reason:
            rows.append({"word": word, "reason": reason})
    return rows


def sort_label_rows(rows: list[dict]) -> list[dict]:
    """Deterministic sort: class, descending weight, family, word."""
    return sorted(
        rows,
        key=lambda r: (r["class"], -float(r["weight"]), r["family"], r["word"]),
    )


def write_labels_csv(rows: list[dict], path: Path) -> None:
    header = ["word", "family", "class", "weight", "strength", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def write_discarded_csv(rows: list[dict], path: Path) -> None:
    header = ["word", "reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def validate(rows: list[dict], vocab_size: int) -> list[str]:
    """Check output invariants; returns a list of error strings."""
    errors: list[str] = []

    seen: set[tuple] = set()
    for row in rows:
        key = (row["word"], row["family"], row["class"])
        if key in seen:
            errors.append(f"duplicate row: {key}")
        seen.add(key)
        if row["word"] != row["word"].lower():
            errors.append(f"non-lowercase word: {row['word']}")
        weight = float(row["weight"])
        if not 0.0 < weight <= 1.0:
            errors.append(f"weight out of range for {row['word']}: {weight}")
        if row["class"] not in CLASSES:
            errors.append(f"unknown class: {row['class']}")

    if len(rows) > vocab_size:
        errors.append("more label rows than input vocabulary words")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the curated functional-keyword CSV from the "
            "top-word vocabulary."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input vocabulary CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    vocab = load_vocabulary(args.input)
    label_rows, labeled = build_label_rows(vocab)
    label_rows = sort_label_rows(label_rows)
    discard_rows = build_discard_rows(vocab, labeled)

    labels_path = args.output_dir / "functional_labels.csv"
    discarded_path = args.output_dir / "functional_labels_discarded.csv"
    write_labels_csv(label_rows, labels_path)
    write_discarded_csv(discard_rows, discarded_path)

    # Summary.
    class_counts: Counter[str] = Counter(r["class"] for r in label_rows)
    unused_rules = sorted(set(KEYWORD_RULES) - set(vocab))

    print("Functional keyword generation complete.")
    print()
    print(f"Input words: {len(vocab)}")
    print(f"Functional keywords: {len(labeled)}")
    print(f"Discarded: {len(discard_rows)}")
    print()
    print("By class:")
    for cls in CLASSES:
        print(f"{cls}: {class_counts.get(cls, 0)}")
    print()
    print(f"Label rows written : {labels_path} ({len(label_rows)} rows)")
    print(f"Discard rows written: {discarded_path} ({len(discard_rows)} rows)")

    if unused_rules:
        print()
        print(
            f"Note: {len(unused_rules)} curated rule words are not present "
            f"in the input vocabulary and were skipped:"
        )
        print("  " + ", ".join(unused_rules))

    errors = validate(label_rows, len(vocab))
    if errors:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)
    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()