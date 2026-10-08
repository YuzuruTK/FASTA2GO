"""Shared text-processing vocabulary and tokenizer for the analysis scripts."""

import re
from pathlib import Path

#: Common English stopwords that carry no categorization signal.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can cannot could
    did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it
    its itself just me more most my myself no nor not now of off on once
    only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they
    this those through to too under until up very was we were what when
    where which while who whom why will with you your yours yourself
    yourselves may might must shall upon within whose
    """.split()
)

#: UniProt annotation metadata tokens (evidence codes, cross-references)
#: that carry no categorization signal.
NOISE_WORDS = frozenset(
    """
    eco pubmed uniprotkb hamap rule mf cc evidence interpro
    prov predicted inferred similarity
    """.split()
)

#: Generic filler words that appear in many descriptions but do not help
#: categorize proteins. Curated exclusion list.
GENERIC_WORDS = frozenset(
    """
    function functions functional protein proteins gene genes product
    products factor factors component components element elements
    molecule molecules entity entities involved involving associated
    associated_with participates participating contributes contributing
    plays play role roles required necessary important critical essential
    acts acting serves serving mediates mediating facilitates
    facilitating helps helping supports supporting putative probable
    predicted possible potential hypothetical unknown uncharacterized
    conserved similar homolog homologous ortholog paralog family member
    contains containing encoding encoded related like type class group
    part form forms forming formation formed complex subunit chain region
    domain domains motif motifs site sites residue residues alpha beta
    gamma delta large small major minor high low long short specific
    general common unique different various multiple single
    thereby thus activity process processes
    """.split()
)

#: All words removed before analysis (stopwords, noise, generic filler).
EXCLUDED = STOPWORDS | NOISE_WORDS | GENERIC_WORDS

#: Token pattern: lowercase alphanumeric words (keeps internal apostrophes).
WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into alphanumeric word tokens."""
    return [word for word in (t.strip("'") for t in WORD_RE.findall(text.lower())) if word]


def is_noise(word: str) -> bool:
    """Return True for words that carry no categorization signal."""
    return word in EXCLUDED or word.isdigit()


def load_word_file(path: str | Path) -> set[str]:
    """Load words from a text file (one per line, '#' starts a comment)."""
    words: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return words
