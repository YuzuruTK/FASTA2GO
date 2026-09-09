#!/usr/bin/env python3
"""Build the ground-truth protein-function taxonomy dictionary.

Generates ``functional_labels.csv`` at the project root with the schema::

    term,category,broad_roles,score,source

The dictionary is a curated, reusable biological taxonomy that can later be
matched against UniProt annotation word counts to classify words into broad
protein-function categories.

Design goals (precision over quantity):

* lowercase, singular canonical terms (multi-word terms joined by ``_``)
* synonym map stored here in code (canonical -> variants)
* hierarchical enzyme taxonomy (EC level-1 classes vs. level-2 families)
* semicolon-joined broad_roles for multi-role terms
* hard exclusion of annotation-quality/uncertainty terms and generic
  boilerplate
* reproducible generation: edit this file and re-run, never hand-edit the CSV
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = PROJECT_ROOT / "functional_labels.csv"

# ---------------------------------------------------------------------------
# Exclusion sets
# ---------------------------------------------------------------------------

#: Annotation-quality / uncertainty terms that MUST NOT appear in the output.
UNCERTAINTY_WORDS = frozenset(
    """
    putative probable predicted possible potential hypothetical unknown
    uncharacterized candidate
    """.split()
)

#: Generic biological boilerplate / filler words (no categorization signal).
GENERIC_WORDS = frozenset(
    """
    function functions functional protein proteins gene genes product
    products factor factors component components element elements molecule
    molecules entity entities involved involving associated participates
    participating contributes contributing plays play role roles required
    necessary important critical essential acts acting serves serving
    mediates mediating facilitates facilitating helps helping supports
    supporting contains containing encoding encoded related like type class
    group part form forms forming formation formed complex subunit chain
    region domain domains motif motifs site sites residue residues
    alpha beta gamma delta large small major minor high low long short
    specific general common unique different various multiple single thereby
    thus activity process processes conserved similar homolog homologous
    ortholog paralog family member
    """.split()
)

#: Pure stopwords (English function words).
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be
    because been before being below between both but by can cannot could did
    do does doing down during each few for from further had has have having
    he her here hers herself him himself his how i if in into is it its
    itself just me more most my myself no nor not now of off on once only or
    other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those
    through to too under until up very was we were what when where which
    while who whom why will with you your yours yourself yourselves may
    might must shall upon within whose
    """.split()
)

EXCLUDED_TERMS = UNCERTAINTY_WORDS | GENERIC_WORDS | STOPWORDS

# ---------------------------------------------------------------------------
# Broad-role taxonomy
# ---------------------------------------------------------------------------

BROAD_ROLES = OrderedDict(
    [
        ("Enzyme", "Enzyme"),
        ("Transporter", "Transporter"),
        ("Receptor", "Receptor"),
        ("Regulatory Protein", "Regulatory Protein"),
        ("DNA-associated Protein", "DNA-associated Protein"),
        ("RNA-associated Protein", "RNA-associated Protein"),
        ("Ribosomal Protein", "Ribosomal Protein"),
        ("Structural Protein", "Structural Protein"),
        ("Motor Protein", "Motor Protein"),
        ("Chaperone", "Chaperone"),
        ("Signaling Protein", "Signaling Protein"),
        ("Defense Protein", "Defense Protein"),
        ("Storage Protein", "Storage Protein"),
        ("Membrane-associated Protein", "Membrane-associated Protein"),
        ("Metabolic Protein", "Metabolic Protein"),
        ("Unknown/Other", "Unknown/Other"),
    ]
)

CAT_FUNCTIONAL = "FUNCTIONAL_LABEL"
CAT_SIGNAL = "BIOLOGICAL_SIGNAL"
CAT_CONTEXTUAL = "CONTEXTUAL"

# ---------------------------------------------------------------------------
# Curated term definitions: (canonical_term, category, broad_roles, score)
# ---------------------------------------------------------------------------

TERMS: list[tuple[str, str, str | tuple[str, ...], int]] = []


def add(category: str, roles: str | tuple[str, ...], score: int, *terms: str) -> None:
    """Register one or more canonical terms under a category + roles."""
    for term in terms:
        TERMS.append((term, category, roles, score))


# ===========================================================================
# ENZYME — Level-1 EC classes
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "oxidoreductase", "transferase", "hydrolase", "lyase", "isomerase",
    "ligase", "translocase")

# ===========================================================================
# ENZYME — Level-2 families (oxidoreductases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "dehydrogenase", "oxidase", "reductase", "oxygenase", "hydroxylase",
    "peroxidase", "catalase", "desaturase", "dioxygenase", "monooxygenase",
    "alcohol_dehydrogenase", "aldehyde_dehydrogenase", "lactate_dehydrogenase",
    "malate_dehydrogenase", "isocitrate_dehydrogenase",
    "glucose_dehydrogenase", "glutamate_dehydrogenase",
    "succinate_dehydrogenase", "glycerol_dehydrogenase",
    "monoamine_oxidase", "diamine_oxidase", "cytochrome_c_oxidase",
    "cytochrome_p450", "cytochrome", "flavoprotein", "ferredoxin",
    "rubredoxin", "thioredoxin", "glutaredoxin", "disulfide_isomerase",
    "laccase", "tyrosinase", "polyphenol_oxidase", "glucose_oxidase",
    "xanthine_oxidase", "xanthine_dehydrogenase", "nitric_oxide_synthase",
    "glutathione_peroxidase", "superoxide_dismutase",
    "ribonucleotide_reductase", "thioredoxin_reductase",
    "glutathione_reductase", "dihydrofolate_reductase",
    "methionine_sulfoxide_reductase", "sulfite_reductase",
    "nitrate_reductase", "nitrite_reductase", "nitrogenase", "hydrogenase",
    "aromatase", "lipoxygenase", "cyclooxygenase", "prostaglandin_synthase",
    "leukotriene_synthase", "aldehyde_oxidase", "sulfite_oxidase",
    "d_amino_acid_oxidase", "l_amino_acid_oxidase", "acyl_coa_dehydrogenase",
    "acyl_coa_oxidase", "hydroxysteroid_dehydrogenase",
    "3_hydroxyacyl_coa_dehydrogenase", "pyruvate_dehydrogenase",
    "alpha_ketoglutarate_dehydrogenase", "branched_chain_ketoacid_dehydrogenase",
    "dihydrolipoamide_dehydrogenase", "dihydropyrimidine_dehydrogenase",
    "quinone_reductase", "nadph_oxidase", "xanthine_oxidase",
    "glutamate_synthase", "formate_dehydrogenase", "carbon_monoxide_dehydrogenase")

# ===========================================================================
# ENZYME — Level-2 families (transferases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "kinase", "phosphotransferase", "aminotransferase", "transaminase",
    "methyltransferase", "acetyltransferase", "glycosyltransferase",
    "glucosyltransferase", "galactosyltransferase", "fucosyltransferase",
    "sialyltransferase", "glucuronosyltransferase", "sulfotransferase",
    "prenyltransferase", "farnesyltransferase", "geranylgeranyltransferase",
    "myristoyltransferase", "palmitoyltransferase", "thymidylate_synthase",
    "glycogen_synthase", "starch_synthase", "chitin_synthase",
    "cellulose_synthase", "hyaluronan_synthase", "nucleotidyltransferase",
    "adenylyltransferase", "guanylyltransferase", "ribosyltransferase",
    "phosphoribosyltransferase", "creatine_kinase", "pyruvate_kinase",
    "hexokinase", "glucokinase", "fructokinase", "phosphofructokinase",
    "phosphoglycerate_kinase", "protein_kinase", "serine_threonine_kinase",
    "tyrosine_kinase", "histidine_kinase",
    "map_kinase", "mitogen_activated_protein_kinase",
    "cyclin_dependent_kinase", "checkpoint_kinase",
    "phosphoinositide_kinase", "phosphatidylinositol_kinase",
    "acetyl_coa_synthetase", "fatty_acid_synthase", "polyketide_synthase",
    "glycogen_phosphorylase", "uracil_phosphoribosyltransferase",
    "hypoxanthine_phosphoribosyltransferase", "adenine_phosphoribosyltransferase",
    "thymidine_kinase", "deoxycytidine_kinase", "nucleoside_kinase",
    "glucosamine_phosphate_acetyltransferase", "histone_acetyltransferase",
    "protein_arginine_methyltransferase", "dna_methyltransferase",
    "rna_methyltransferase", "glutathione_s_transferase", "choline_acetyltransferase")

# ===========================================================================
# ENZYME — Level-2 families (hydrolases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "phosphatase", "protease", "peptidase", "nuclease", "glycosidase",
    "lipase", "esterase", "atpase", "gtpase", "helicase", "polymerase",
    "deaminase", "deacetylase", "deubiquitinase", "phospholipase",
    "sphingomyelinase", "ceramidase", "acetylcholinesterase",
    "butyrylcholinesterase", "cholinesterase", "epoxide_hydrolase",
    "urease", "arginase", "adenosine_deaminase", "cytidine_deaminase",
    "guanine_deaminase", "nucleotidase", "pyrophosphatase",
    "phosphoprotein_phosphatase", "serine_threonine_phosphatase",
    "tyrosine_phosphatase", "lactamase", "beta_lactamase", "penicillinase",
    "cephalosporinase", "aminopeptidase", "carboxypeptidase",
    "endopeptidase", "exopeptidase", "dipeptidase", "tripeptidase",
    "oligopeptidase", "signal_peptidase", "metalloprotease",
    "serine_protease", "cysteine_protease", "aspartic_protease",
    "threonine_protease", "secretase", "gamma_secretase", "beta_secretase",
    "caspase", "calpain", "cathepsin", "trypsin", "chymotrypsin", "elastase",
    "pepsin", "renin", "subtilisin", "thermolysin", "collagenase",
    "gelatinase", "stromelysin", "matrilysin", "matrix_metalloproteinase",
    "thioesterase", "rhodanese", "lysozyme", "hyaluronidase",
    "neuraminidase", "amylase", "cellulase", "xylanase", "chitinase",
    "glucanase", "galactosidase", "glucosidase", "mannosidase",
    "fucosidase", "sialidase", "sucrase", "lactase", "maltase",
    "phospholipase_a2", "phospholipase_c", "phospholipase_d",
    "diacylglycerol_lipase", "monoacylglycerol_lipase",
    "lipoprotein_lipase", "hormone_sensitive_lipase",
    "phosphatidylinositol_phospholipase", "inositol_phosphatase",
    "adenosine_triphosphatase", "synaptotagmin", "dynamin", "plasmin",
    "thrombin", "coagulation_factor", "fibrinolysin", "dicer", "drosha",
    "argonaut", "piwi", "ribonuclease", "endonuclease", "exonuclease",
    "flap_endonuclease", "ap_endonuclease", "glycosylase", "dna_glycosylase",
    "uracil_dna_glycosylase", "restriction_enzyme")

# ===========================================================================
# ENZYME — Level-2 families (lyases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "aldolase", "dehydratase", "decarboxylase", "cyclase", "adenylate_cyclase",
    "guanylate_cyclase", "fumarase", "enolase", "carbonic_anhydrase",
    "fumarate_hydratase", "aconitase", "citrate_lyase", "isocitrate_lyase",
    "malate_synthase", "fructose_bisphosphate_aldolase",
    "threonine_aldolase", "serine_dehydratase", "threonine_dehydratase",
    "histidine_decarboxylase", "ornithine_decarboxylase",
    "arginine_decarboxylase", "glutamate_decarboxylase",
    "tyrosine_decarboxylase", "dopa_decarboxylase",
    "porphobilinogen_synthase", "aminolevulinic_acid_synthase",
    "phenylalanine_ammonia_lyase", "tyrosine_ammonia_lyase",
    "histidine_ammonia_lyase", "arogenate_dehydratase",
    "dihydroxy_acid_dehydratase", "deoxyribose_phosphate_aldolase")

# ===========================================================================
# ENZYME — Level-2 families (isomerases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "topoisomerase", "epimerase", "racemase", "mutase", "photoisomerase",
    "triosephosphate_isomerase", "phosphoglucose_isomerase",
    "phosphoglycerate_mutase", "phosphoglucomutase",
    "bisphosphoglycerate_mutase", "methylmalonyl_coa_mutase",
    "dna_topoisomerase", "topoisomerase_i", "topoisomerase_ii",
    "peptidylprolyl_isomerase", "cyclophilin", "fkbp", "parvulin",
    "retinal_isomerase", "phosphoglucosamine_mutase", "uroporphyrinogen_synthase")

# ===========================================================================
# ENZYME — Level-2 families (ligases)
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "synthetase", "synthase", "carboxylase", "recombinase", "dna_ligase",
    "rna_ligase", "aminoacyl_trna_synthetase", "alanyl_trna_synthetase",
    "arginyl_trna_synthetase", "asparaginyl_trna_synthetase",
    "aspartyl_trna_synthetase", "cysteinyl_trna_synthetase",
    "glutamyl_trna_synthetase", "glutaminyl_trna_synthetase",
    "glycyl_trna_synthetase", "histidyl_trna_synthetase",
    "isoleucyl_trna_synthetase", "leucyl_trna_synthetase",
    "lysyl_trna_synthetase", "methionyl_trna_synthetase",
    "phenylalanyl_trna_synthetase", "prolyl_trna_synthetase",
    "seryl_trna_synthetase", "threonyl_trna_synthetase",
    "tryptophanyl_trna_synthetase", "tyrosyl_trna_synthetase",
    "valyl_trna_synthetase", "acetyl_coa_carboxylase",
    "propionyl_coa_carboxylase", "pyruvate_carboxylase",
    "glutamine_synthetase", "asparagine_synthetase",
    "glutathione_synthetase", "folate_synthetase",
    "succinyl_coa_synthetase", "carbamoyl_phosphate_synthetase",
    "ubiquitin_conjugating_enzyme", "ubiquitin_ligase", "sumo_ligase")

# ===========================================================================
# ENZYME — translocases / ATP synthases
# ===========================================================================
add(CAT_FUNCTIONAL, "Enzyme", 10,
    "atp_synthase", "f0f1_atp_synthase", "sec_translocase", "tat_translocase",
    "mitochondrial_translocase", "adenine_nucleotide_translocase",
    "phosphate_carrier", "atpase")

# ===========================================================================
# TRANSPORT
# ===========================================================================
add(CAT_FUNCTIONAL, "Transporter", 10,
    "transporter", "channel", "carrier", "pump", "permease", "symporter",
    "antiporter", "translocase", "efflux", "importer", "exporter",
    "aquaporin", "porin", "ion_channel", "anion_channel", "cation_channel",
    "potassium_channel", "sodium_channel", "calcium_channel",
    "chloride_channel", "voltage_gated", "ligand_gated",
    "atp_binding_cassette", "abc_transporter", "solute_carrier",
    "glucose_transporter", "amino_acid_transporter", "peptide_transporter",
    "nucleotide_transporter", "phosphate_transporter", "iron_transporter",
    "zinc_transporter", "copper_transporter", "membrane_transport",
    "major_facilitator_superfamily", "sodium_potassium_atpase",
    "calcium_atpase", "proton_atpase", "v_atpase", "f_atpase", "p_type_atpase",
    "sodium_potassium_pump", "proton_pump", "multidrug", "multidrug_resistance",
    "drug_efflux", "xenobiotic_transporter", "organic_anion_transporter",
    "organic_cation_transporter", "urea_transporter", "water_channel",
    "glycerol_channel", "maltose_transporter", "lactose_permease",
    "galactose_transporter", "fructose_transporter", "mannose_transporter",
    "nucleoside_transporter", "vitamin_transporter", "bile_acid_transporter",
    "fatty_acid_transporter", "monocarboxylate_transporter",
    "dicarboxylate_transporter", "tricarboxylate_transporter",
    "potassium_transporter", "sodium_transporter", "magnesium_transporter",
    "manganese_transporter", "anion_exchanger", "bicarbonate_transporter",
    "chloride_bicarbonate_exchanger", "mitochondrial_carrier",
    "uncoupling_protein", "voltage_dependent_anion_channel",
    "glutamate_transporter", "glycine_transporter", "dopamine_transporter",
    "serotonin_transporter", "norepinephrine_transporter", "choline_transporter",
    "taurine_transporter", "proline_transporter", "betaine_transporter")

# ===========================================================================
# RECEPTOR
# ===========================================================================
add(CAT_FUNCTIONAL, ("Receptor", "Signaling Protein"), 10,
    "receptor", "chemokine_receptor", "cytokine_receptor",
    "g_protein_coupled_receptor", "toll_like_receptor",
    "pattern_recognition_receptor", "nuclear_receptor", "steroid_receptor",
    "ionotropic_receptor", "metabotropic_receptor", "scavenger_receptor",
    "nod_like_receptor", "rig_like_receptor", "t_cell_receptor",
    "b_cell_receptor", "immunoglobulin_receptor", "fc_receptor",
    "growth_factor_receptor", "insulin_receptor",
    "insulin_like_growth_factor_receptor", "epidermal_growth_factor_receptor",
    "fibroblast_growth_factor_receptor",
    "platelet_derived_growth_factor_receptor",
    "vascular_endothelial_growth_factor_receptor",
    "nerve_growth_factor_receptor", "transforming_growth_factor_receptor",
    "tumor_necrosis_factor_receptor", "death_receptor", "fas_receptor",
    "tnf_receptor", "dopamine_receptor", "serotonin_receptor",
    "histamine_receptor", "acetylcholine_receptor", "nicotinic_receptor",
    "muscarinic_receptor", "gaba_receptor", "glutamate_receptor",
    "nmda_receptor", "ampa_receptor", "kainate_receptor", "glycine_receptor",
    "adrenergic_receptor", "beta_adrenergic_receptor",
    "alpha_adrenergic_receptor", "opioid_receptor", "cannabinoid_receptor",
    "rhodopsin", "opsin", "integrin", "cadherin", "selectin", "lectin",
    "mannose_receptor", "transferrin_receptor", "ldl_receptor",
    "lipoprotein_receptor")

add(CAT_FUNCTIONAL, ("Receptor", "Signaling Protein", "Enzyme"), 10,
    "receptor_kinase", "receptor_tyrosine_kinase")

add(CAT_FUNCTIONAL, ("Receptor", "Signaling Protein"), 9,
    "cell_adhesion", "adhesion", "adhesin")

# ===========================================================================
# REGULATORY PROTEIN
# ===========================================================================
add(CAT_FUNCTIONAL, "Regulatory Protein", 10,
    "regulator", "activator", "repressor", "sigma", "enhancer", "silencer",
    "response_regulator", "transcription_factor", "coactivator", "corepressor",
    "transcriptional", "transcription", "replication_factor",
    "elongation_factor", "initiation_factor", "termination_factor",
    "rho_factor", "sigma_factor", "anti_sigma", "anti_terminator",
    "homeobox", "homeodomain", "zinc_finger", "leucine_zipper",
    "helix_loop_helix", "helix_turn_helix", "winged_helix", "forkhead",
    "pax", "hox", "pou_domain", "bhlh", "bzip", "ets_domain", "sox", "t_box",
    "smad", "stat", "nf_kb", "ap_1", "creb", "nuclear_factor", "mediator",
    "tata_binding_protein", "tfiid", "tfii", "general_transcription_factor",
    "basal_transcription_factor", "histone_acetyltransferase",
    "histone_deacetylase", "histone_methyltransferase", "histone_demethylase",
    "dna_methyltransferase", "methyl_cpg_binding", "chromatin_remodeling",
    "swi_snf", "polycomb", "trithorax", "promoter", "operator", "terminator",
    "riboswitch", "two_component", "sensor", "repression", "activation")

# ===========================================================================
# DNA-ASSOCIATED
# ===========================================================================
add(CAT_SIGNAL, "DNA-associated Protein", 8,
    "dna", "replication", "repair", "recombination", "chromosome",
    "chromatin", "telomere", "nucleoid", "histone", "nucleosome",
    "centromere", "dna_binding", "dna_replication", "dna_repair",
    "dna_recombination", "replication_origin", "dna_damage",
    "base_excision", "nucleotide_excision", "mismatch_repair",
    "homologous_recombination", "transposon", "plasmid", "gyrase", "primase",
    "dna_ligase", "single_strand_binding", "ap_endonuclease", "glycosylase",
    "dna_glycosylase", "uracil_dna_glycosylase", "alkyltransferase",
    "photolyase", "resolvase", "holliday_junction", "rad51", "reca", "muts",
    "mutl", "muth", "transposase", "integrase", "dna_methylase",
    "origin_recognition_complex", "clamp", "sliding_clamp",
    "proliferating_cell_nuclear_antigen", "replication_protein_a",
    "minichromosome_maintenance", "cdt1", "cdc6", "cohesin",
    "condensin", "smc", "telomerase", "reverse_transcriptase")

# ===========================================================================
# RNA-ASSOCIATED
# ===========================================================================
add(CAT_SIGNAL, "RNA-associated Protein", 8,
    "rna", "rrna", "trna", "mrna", "ribonuclease", "spliceosome", "splicing",
    "translation", "ribosome", "ribozyme", "polyadenylation", "rna_binding",
    "rna_processing", "rna_splicing", "snrna", "snorna", "mirna", "sirna",
    "ncrna", "pseudouridine", "rna_polymerase", "rna_helicase",
    "rna_methyltransferase", "capping", "rna_polymerase_i",
    "rna_polymerase_ii", "rna_polymerase_iii",
    "small_nuclear_ribonucleoprotein", "snrnp", "splice_factor",
    "alternative_splicing", "exon_junction", "nonsense_mediated_decay",
    "rna_editing", "adenosine_deaminase_rna", "capping_enzyme",
    "poly_a_polymerase", "deadenylase", "decapping", "exosome", "rnase",
    "rnase_h", "rna_interference", "rna_silencing", "rna_induced_silencing",
    "ribonuclease_p", "pseudouridine_synthase", "trna_synthetase",
    "rna_ligase", "translation_initiation", "translation_elongation")

# ===========================================================================
# RIBOSOMAL
# ===========================================================================
add(CAT_FUNCTIONAL, "Ribosomal Protein", 10,
    "ribosomal", "ribosome", "large_subunit", "small_subunit",
    "ribosomal_protein", "40s", "60s", "30s", "50s", "80s", "70s",
    "ribosomal_rna", "16s", "18s", "23s", "28s", "5s", "5.8s")

# ===========================================================================
# STRUCTURAL
# ===========================================================================
add(CAT_FUNCTIONAL, "Structural Protein", 9,
    "structural", "collagen", "keratin", "actin", "tubulin", "cytoskeleton",
    "scaffold", "filament", "microtubule", "microfilament",
    "intermediate_filament", "lamin", "vimentin", "fibronectin", "elastin",
    "fibrin", "fibroin", "flagellin", "pilin", "capsid", "nucleocapsid",
    "matrix", "extracellular_matrix", "cell_wall", "tropocollagen",
    "procollagen", "tropomyosin", "troponin", "titin", "nebulin", "desmin",
    "spectrin", "ankyrin", "dystrophin", "fibrillin", "laminin", "nidogen",
    "perlecan", "aggrecan", "decorin", "proteoglycan", "glycosaminoglycan",
    "hyaluronan", "chitin", "cellulose", "pectin", "fimbrial", "pili",
    "pilus", "flagellum", "cilia", "centrosome", "centriole", "kinetochore",
    "spindle", "septin", "nuclear_pore", "nucleoporin", "clathrin",
    "coatomer", "copi", "copii", "caveolin", "ezrin", "radixin", "moesin",
    "vinculin", "talin", "paxillin", "focal_adhesion", "tight_junction",
    "adherens_junction", "gap_junction", "desmosome", "hemidesmosome",
    "occludin", "claudin", "connexin", "gel", "mucus", "mucin", "spore",
    "spore_coat", "enamel", "amelogenin")

# ===========================================================================
# MOTOR
# ===========================================================================
add(CAT_FUNCTIONAL, "Motor Protein", 9,
    "myosin", "kinesin", "dynein", "motor", "contractile", "flagellar",
    "ciliary", "axonemal", "actin_binding", "myosin_i", "myosin_ii",
    "myosin_v", "axonemal_dynein", "cytoplasmic_dynein", "dynein_heavy_chain",
    "dynein_light_chain", "dynein_intermediate_chain", "prestin", "dynamin")

# ===========================================================================
# CHAPERONE
# ===========================================================================
add(CAT_FUNCTIONAL, "Chaperone", 9,
    "chaperone", "hsp", "hsp70", "hsp90", "folding", "refolding", "groel",
    "groes", "chaperonin", "heat_shock", "heat_shock_protein", "cochaperone",
    "protein_folding", "foldase", "hsp40", "hsp60", "hsp100", "dnaj", "dnak",
    "htpg", "clpb", "clpa", "clpx", "small_heat_shock_protein",
    "alpha_crystallin", "hspb", "secb", "trigger_factor", "calnexin",
    "calreticulin", "pdi", "protein_disulfide_isomerase", "erp57", "bip",
    "grp78", "grp94")

# ===========================================================================
# SIGNALING
# ===========================================================================
add(CAT_SIGNAL, "Signaling Protein", 8,
    "signal", "signaling", "sensor", "messenger", "cytokine", "hormone",
    "growth_factor", "phosphorelay", "quorum", "second_messenger", "camp",
    "cgmp", "calmodulin", "g_protein", "gtpase", "kinase", "phosphatase",
    "chemotaxis", "neurotransmitter", "ligand", "agonist", "antagonist",
    "adrenergic", "insulin", "glucagon", "epidermal_growth_factor",
    "fibroblast_growth_factor", "interferon", "interleukin",
    "tumor_necrosis_factor", "heterotrimeric_g_protein", "small_gtpase",
    "ras", "rho", "rac", "cdc42", "rab", "ran", "arf", "galpha", "gbeta",
    "ggamma", "guanine_nucleotide_exchange_factor",
    "gtpase_activating_protein", "guanine_nucleotide_dissociation_inhibitor",
    "phosphodiesterase", "phospholipase_c", "diacylglycerol",
    "inositol_trisphosphate", "protein_kinase_a", "protein_kinase_c",
    "protein_kinase_g", "calcineurin", "calcium", "phosphatidylinositol",
    "pip2", "pip3", "akt", "pkb", "pi3k", "mtor", "mapk", "erk", "jnk",
    "p38", "wnt", "notch", "hedgehog", "sonic_hedgehog", "tgf_beta", "bmp",
    "fgf", "egf", "vegf", "pdgf", "ngf", "ghrelin", "leptin", "adiponectin",
    "resistin", "gh", "prolactin", "oxytocin", "vasopressin", "melatonin",
    "serotonin", "dopamine", "norepinephrine", "epinephrine", "histamine",
    "acetylcholine", "gaba", "glycine", "glutamate", "nitric_oxide",
    "carbon_monoxide", "hydrogen_sulfide", "prostacyclin", "thromboxane",
    "leukotriene", "prostaglandin", "corticosterone", "cortisol",
    "aldosterone", "estrogen", "progesterone", "testosterone", "androgen",
    "thyroid_hormone", "retinoic_acid", "vitamin_d", "eicosanoid",
    "chemoattractant", "chemorepellent", "pheromone", "chemokine",
    "lymphokine", "monokine", "transforming_growth_factor",
    "colony_stimulating_factor", "erythropoietin", "thrombopoietin",
    "stem_cell_factor", "signal_transduction", "second_messenger")

# ===========================================================================
# DEFENSE
# ===========================================================================
add(CAT_FUNCTIONAL, "Defense Protein", 9,
    "antibody", "immune", "antigen", "complement", "toxin", "antimicrobial",
    "defense", "virulence", "immunoglobulin", "interferon", "lysozyme",
    "defensin", "antiviral", "antibacterial", "antifungal", "restriction",
    "phage", "bacteriocin", "resistance", "immunoglobulin_g", "iga", "igm",
    "ige", "igd", "heavy_chain", "light_chain", "fab", "fc", "lactoferrin",
    "transferrin", "cathelicidin", "dermcidin", "histatin", "thrombocidin",
    "colicin", "microcin", "pyocin", "nisin", "plantaricin", "antitoxin",
    "endotoxin", "exotoxin", "hemolysin", "enterotoxin", "neurotoxin",
    "cytolysin", "leukocidin", "pertussis_toxin", "cholera_toxin",
    "diphtheria_toxin", "tetanus_toxin", "botulinum_toxin", "shiga_toxin",
    "ricin", "abrin", "saporin", "immunotoxin", "adhesin", "invasin",
    "capsule", "type_iii_secretion", "type_iv_secretion", "effector",
    "avirulence", "epitope", "mhc", "major_histocompatibility", "hla",
    "crispr", "cas9", "cas3", "restriction_modification", "complement_component",
    "c3", "c5", "immunity", "self_nonself", "opsonin", "phagocytosis")

# ===========================================================================
# STORAGE
# ===========================================================================
add(CAT_FUNCTIONAL, "Storage Protein", 9,
    "ferritin", "storage", "reservoir", "heme_storage", "iron_storage",
    "casein", "ovalbumin", "zein", "gliadin", "gluten", "seed_storage",
    "metal_storage", "calcium_storage", "hemosiderin", "metallothionein",
    "phytochelatin", "calsequestrin", "osteocalcin", "osteopontin",
    "bone_sialoprotein", "beta_casein", "kappa_casein", "lactoglobulin",
    "lactalbumin", "ovotransferrin", "vitellogenin", "legumin", "vicilin",
    "glycinin", "conglycinin", "prolamin", "glutelin", "albumin",
    "serum_albumin", "caspase")

# ===========================================================================
# MEMBRANE-ASSOCIATED
# ===========================================================================
add(CAT_SIGNAL, "Membrane-associated Protein", 7,
    "membrane", "transmembrane", "outer_membrane", "inner_membrane",
    "lipoprotein", "glycoprotein", "periplasmic", "integral_membrane",
    "peripheral_membrane", "lipid_anchor", "gpi_anchor", "secreted",
    "extracellular", "cell_surface", "envelope", "capsule", "lipid_raft",
    "caveola", "endosome", "lysosome", "peroxisome", "vacuole", "vesicle",
    "thylakoid", "chloroplast", "mitochondrion", "outer_envelope",
    "inner_envelope", "plasma_membrane")

# ===========================================================================
# METABOLIC / CONTEXTUAL
# ===========================================================================
add(CAT_CONTEXTUAL, "Metabolic Protein", 6,
    "metabolism", "biosynthesis", "catabolism", "glycolysis",
    "gluconeogenesis", "citrate", "tca", "pentose_phosphate",
    "fatty_acid", "amino_acid", "nucleotide", "carbon_fixation",
    "nitrogen_fixation", "photosynthesis", "respiration", "fermentation",
    "electron_transport", "oxidative_phosphorylation", "urea_cycle", "purine",
    "pyrimidine", "cholesterol", "steroid", "lipid", "carbohydrate",
    "polysaccharide", "peptidoglycan", "lipogenesis", "beta_oxidation",
    "lipolysis", "ketogenesis", "ketolysis", "cholesterol_biosynthesis",
    "steroidogenesis", "bile_acid_biosynthesis", "heme_biosynthesis",
    "porphyrin", "chlorophyll", "carotenoid", "isoprenoid", "terpenoid",
    "alkaloid", "flavonoid", "lignin_biosynthesis", "cell_wall_biosynthesis",
    "peptidoglycan_biosynthesis", "autotrophy", "methylotrophy",
    "methanogenesis", "sulfur_metabolism", "nitrogen_metabolism",
    "carbon_metabolism", "amino_acid_metabolism", "nucleotide_metabolism",
    "fatty_acid_metabolism", "lipid_metabolism", "carbohydrate_metabolism",
    "energy_metabolism", "redox", "electron_transfer", "proton_transport",
    "chemiosmosis", "photophosphorylation", "calvin_cycle", "krebs_cycle",
    "glycogen_metabolism", "starch_metabolism", "cell_cycle", "mitosis",
    "meiosis", "cytokinesis", "cell_division", "proliferation",
    "differentiation", "development", "morphogenesis", "apoptosis",
    "autophagy", "necrosis", "senescence", "circadian", "homeostasis",
    "detoxification", "biotransformation", "conjugation", "glucuronidation",
    "sulfation", "methylation", "acetylation", "phosphorylation",
    "dephosphorylation", "ubiquitination", "sumoylation", "neddylation",
    "prenylation", "lipid_modification", "glycosylation", "proteolysis",
    "proteasome", "ubiquitin_proteasome", "secretion", "exocytosis",
    "endocytosis", "transcytosis", "vesicular_transport",
    "membrane_trafficking", "gene_expression", "gene_regulation",
    "epigenetic", "imprinting", "x_inactivation", "dosage_compensation")

add(CAT_CONTEXTUAL, "Metabolic Protein", 5,
    "pathway", "degradation", "degradative", "anabolic", "katabolic")

# ===========================================================================
# BIOLOGICAL SIGNALS / COFACTORS
# ===========================================================================
add(CAT_SIGNAL, ("Enzyme", "Metabolic Protein"), 8,
    "atp", "gtp", "adp", "amp", "gdp", "nad", "nadh", "nadp", "nadph",
    "fad", "fadh2", "heme", "coenzyme", "cofactor", "flavin", "biotin",
    "folate", "pyridoxal", "thiamine", "cobalamin", "molybdopterin",
    "udp", "udp_glucose", "ctp", "utp", "ttp", "datp", "dgtp", "dctp",
    "dttp", "cdi_gmp", "ppgpp", "coa", "acetyl_coa", "coenzyme_a",
    "succinyl_coa", "malonyl_coa", "thiamin", "thiamine_pyrophosphate",
    "pyridoxal_phosphate", "pyridoxamine", "tetrahydrofolate",
    "dihydrofolate", "s_adenosylmethionine", "sam", "tetrahydrobiopterin",
    "lipoic_acid", "lipoamide", "pantothenic_acid", "riboflavin", "niacin",
    "niacinamide", "ascorbic_acid", "retinol", "retinal", "tocopherol",
    "phylloquinone", "menaquinone", "ubiquinone", "coenzyme_q",
    "plastoquinone", "cytochrome_c", "fe_s_cluster", "iron_sulfur",
    "2fe2s", "4fe4s", "siroheme", "tungsten", "vanadium")

add(CAT_SIGNAL, "Metabolic Protein", 7,
    "iron", "zinc", "copper", "oxygen", "phosphate", "sulfur", "magnesium",
    "manganese", "calcium", "potassium", "sodium", "selenium", "cobalt",
    "nickel", "molybdenum")

# ===========================================================================
# Synonym map: canonical_term -> acceptable variants (space forms,
# abbreviations, alternate spellings). Canonical term is the underscore form.
# Preserved in the generator for downstream matching.
# ===========================================================================

SYNONYMS: dict[str, list[str]] = {
    "transporter": ["transporter", "transport"],
    "atpase": ["atpase", "atp_ase"],
    "response_regulator": ["response_regulator", "response regulator"],
    "iron_storage": ["iron_storage", "iron storage"],
    "growth_factor": ["growth_factor", "growth factor"],
    "large_subunit": ["large_subunit", "large subunit"],
    "small_subunit": ["small_subunit", "small subunit"],
    "outer_membrane": ["outer_membrane", "outer membrane"],
    "inner_membrane": ["inner_membrane", "inner membrane"],
    "heat_shock_protein": ["heat_shock_protein", "heat shock protein"],
    "transcription_factor": ["transcription_factor", "transcription factor"],
    "dna_polymerase": ["dna_polymerase", "dna polymerase"],
    "rna_polymerase": ["rna_polymerase", "rna polymerase"],
    "dna_binding": ["dna_binding", "dna binding"],
    "rna_binding": ["rna_binding", "rna binding"],
    "g_protein_coupled_receptor": [
        "g_protein_coupled_receptor",
        "gpcr",
        "g-protein coupled receptor",
    ],
    "toll_like_receptor": ["toll_like_receptor", "tlr"],
    "atp_binding_cassette": ["atp_binding_cassette", "abc transporter"],
    "amino_acid_transporter": ["amino_acid_transporter", "amino acid transporter"],
    "extracellular_matrix": ["extracellular_matrix", "extracellular matrix"],
    "intermediate_filament": ["intermediate_filament", "intermediate filament"],
    "cell_wall": ["cell_wall", "cell wall"],
    "heat_shock": ["heat_shock", "heat shock"],
    "second_messenger": ["second_messenger", "second messenger"],
    "g_protein": ["g_protein", "g protein"],
    "pattern_recognition_receptor": [
        "pattern_recognition_receptor",
        "pattern recognition receptor",
    ],
    "phosphorelay": ["phosphorelay", "two-component"],
    "receptor_tyrosine_kinase": ["receptor_tyrosine_kinase", "rtk"],
    "major_facilitator_superfamily": ["major_facilitator_superfamily", "mfs transporter"],
    "atp_synthase": ["atp_synthase", "atp synthase"],
    "mitochondrial_carrier": ["mitochondrial_carrier", "mitochondrial carrier"],
    "protein_disulfide_isomerase": ["protein_disulfide_isomerase", "pdi"],
    "ubiquitin_ligase": ["ubiquitin_ligase", "ubiquitin ligase"],
}


def normalize(term: str) -> str:
    """Normalize a term: lowercase, strip, collapse whitespace, underscores."""
    term = term.strip().lower()
    term = re.sub(r"\s+", "_", term)
    return term


def roles_str(roles: str | tuple[str, ...]) -> str:
    """Convert a single role string or tuple/list to a ';'-joined string."""
    if isinstance(roles, str):
        return roles
    return ";".join(roles)


def build_records() -> tuple[list[dict], dict, Counter, Counter]:
    """Assemble, normalize, dedupe, and filter term records.

    Returns (records, duplicate_info, category_counts, excluded_counts).
    """
    seen: dict[str, dict] = {}
    duplicates: dict[str, list[str]] = {}
    excluded_counts: Counter = Counter()

    for term, category, roles, score in TERMS:
        canonical = normalize(term)
        if not canonical:
            continue
        if canonical in EXCLUDED_TERMS:
            excluded_counts[canonical] += 1
            continue
        # Drop terms whose sub-words are uncertainty/quality markers
        # (e.g. hypothetical_x, putative_y). We intentionally do NOT drop
        # compound terms merely because a generic sub-word appears
        # (e.g. large_subunit), as those are legitimate biological entities.
        if any(word in UNCERTAINTY_WORDS for word in canonical.split("_")):
            excluded_counts[canonical] += 1
            continue

        record = {
            "term": canonical,
            "category": category,
            "broad_roles": roles_str(roles),
            "score": score,
            "source": "manual",
        }
        if canonical in seen:
            duplicates.setdefault(canonical, []).append(
                f"{seen[canonical]['category']}/{seen[canonical]['broad_roles']}/"
                f"{seen[canonical]['score']}"
            )
            # Keep the higher-score definition; on tie, prefer the first.
            if score > seen[canonical]["score"]:
                seen[canonical] = record
        else:
            seen[canonical] = record

    records = list(seen.values())
    # Stable sort by category, then broad_roles, then term.
    records.sort(key=lambda r: (r["category"], r["broad_roles"], r["term"]))

    category_counts = Counter(r["category"] for r in records)
    return records, duplicates, category_counts, excluded_counts


def write_csv(records: list[dict], path: Path) -> None:
    """Write records to CSV with the required schema."""
    header = ["term", "category", "broad_roles", "score", "source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def validate(records: list[dict], path: Path) -> None:
    """Assert output invariants; exit non-zero on failure."""
    errors: list[str] = []

    if not (1000 <= len(records) <= 3000):
        errors.append(f"row count {len(records)} outside target 1000-3000")

    terms = [r["term"] for r in records]
    if len(terms) != len(set(terms)):
        errors.append("duplicate canonical terms present")

    for record in records:
        term = record["term"]
        if term != term.lower():
            errors.append(f"non-lowercase term: {term}")
        if term in EXCLUDED_TERMS:
            errors.append(f"excluded word present as standalone term: {term}")
        if any(word in UNCERTAINTY_WORDS for word in term.split("_")):
            errors.append(f"uncertainty marker present in term: {term}")

    if errors:
        print("\nVALIDATION FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print(f"\nVALIDATION PASSED: {len(records)} records in {path}")


def main() -> None:
    records, duplicates, category_counts, excluded_counts = build_records()
    write_csv(records, OUTPUT_PATH)

    # Statistics.
    print("=" * 60)
    print("  functional_labels.csv — generation summary")
    print("=" * 60)
    print(f"  Output file       : {OUTPUT_PATH}")
    print(f"  Total terms       : {len(records):,}")
    print()
    print("  Category counts:")
    for category, count in sorted(category_counts.items()):
        print(f"    {category:<22} {count:>5}")
    print()
    print("  Broad-role counts:")
    role_counter: Counter = Counter()
    for record in records:
        for role in record["broad_roles"].split(";"):
            role_counter[role.strip()] += 1
    for role in BROAD_ROLES:
        print(f"    {role:<26} {role_counter.get(role, 0):>5}")
    print()
    print(f"  Duplicate removals: {sum(len(v) for v in duplicates.values()):,}")
    for term, dupes in sorted(duplicates.items()):
        print(f"    {term}: kept ({dupes[0]}), dropped {len(dupes)}")
    print()
    print(f"  Excluded stopwords/boilerplate: {sum(excluded_counts.values()):,}")

    validate(records, OUTPUT_PATH)


if __name__ == "__main__":
    main()