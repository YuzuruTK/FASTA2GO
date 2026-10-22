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
    where which while who whom why will with without within you your yours
    yourself yourselves may might must shall upon whose
    """.split()
)

#: Spelled-out numbers carry no categorization signal on their own.
NUMBER_WORDS = frozenset(
    """
    one two three four five six seven eight nine ten
    """.split()
)

#: Fragments left behind when hyphenated compounds are split
#: ("pre-mRNA" -> "pre", "non-catalytic" -> "non"). They are prefixes,
#: connectives or vague prose, never keywords by themselves.
FRAGMENT_WORDS = frozenset(
    """
    act cis de end ends key non per post pre trans versus vs via etc
    eg ie other others respectively thereof therein hereby herein
    """.split()
)

#: Roman numerals used as generic designators ("complex II", "type IV").
#: Single letters are already rejected by the length rule below.
ROMAN_NUMERALS = frozenset(
    """
    ii iii iv vi vii viii ix xi xii
    """.split()
)

#: Two-character tokens that do carry real signal. Anything else this
#: short is notation debris from the corpus: chemical position labels
#: (N6, C1), subunit/gene abbreviations (S5, L1, NF, ER) or numeric codes
#: (9Z, 2C). Keeping short tokens only by explicit whitelist is safer than
#: guessing with a pattern.
SHORT_TOKENS = frozenset(
    """
    aa ca cl co cu ec fe li mg mn mo na ni ph se si zn
    e1 e2 e3 h1 h2 h3 h4 b6 b12
    5s 8s 16s 18s 23s 25s 28s 30s 40s 45s 48s 50s 54s 60s 70s 80s 90s
    """.split()
)

#: UniProt annotation metadata tokens (evidence codes, cross-references)
#: that carry no categorization signal.
NOISE_WORDS = frozenset(
    """
    eco pubmed uniprotkb hamap rule mf cc evidence interpro go
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

#: All words removed before analysis (stopwords, numbers, fragments,
#: annotation noise, generic filler).
EXCLUDED = STOPWORDS | NUMBER_WORDS | FRAGMENT_WORDS | NOISE_WORDS | GENERIC_WORDS

#: Words always removed from phrases: grammatical/structural debris only.
#: Generic biology words ("protein", "activity", "subunit") are kept here
#: because inside a concept they often carry the distinction: dropping them
#: turns "identical protein binding" into "identical_binding".
PHRASE_EXCLUDED = STOPWORDS | NUMBER_WORDS | FRAGMENT_WORDS | NOISE_WORDS

#: Token pattern: lowercase alphanumeric words (keeps internal apostrophes).
WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase a text and split it into alphanumeric word tokens."""
    return [word for word in (t.strip("'") for t in WORD_RE.findall(text.lower())) if word]


def _structural_noise(word: str) -> bool:
    """True for words that are never part of a concept, phrase or not."""
    if word in PHRASE_EXCLUDED:
        return True
    if len(word) == 1:
        return True
    if word.isdigit():
        return True
    if word in ROMAN_NUMERALS:
        return True
    return len(word) == 2 and word not in SHORT_TOKENS


def is_noise(word: str) -> bool:
    """Return True for words that carry no categorization signal.

    Applies the exclusion vocabulary (including generic filler), then the
    structural rules: single characters (every isolated letter is notation
    debris: "C atoms", "L-alanine", "S5"), multi-digit numbers, roman-numeral
    designators ("complex II"), and any two-character token outside the
    curated whitelist.
    """
    return word in GENERIC_WORDS or _structural_noise(word)


def is_phrase_noise(word: str) -> bool:
    """Like :func:`is_noise`, but keeps generic filler words.

    Used when normalizing a multi-word concept so the phrase stays faithful
    to its source ("structural constituent of ribosome", "identical protein
    binding"). Only grammatical and notation debris is dropped.
    """
    return _structural_noise(word)


def load_word_file(path: str | Path) -> set[str]:
    """Load words from a text file (one per line, '#' starts a comment)."""
    words: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        if line:
            words.add(line)
    return words
