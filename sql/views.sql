-- Derived views. Nothing here is stored twice: the chemical classes are sums of
-- the stored composition percentages, and the summary counts join the GO table.

CREATE OR REPLACE VIEW protein_features AS
SELECT
    p.source_file,
    p.fasta_id,
    p.uniprot_accession,
    p.sequence_length,
    f.feature_version,
    f.canonical_length,
    f.noncanonical_residues,
    f.molecular_weight,
    f.isoelectric_point,
    f.gravy,
    f.instability_index,
    f.aliphatic_index,
    f.aromaticity,
    f.sequence_entropy,
    a.* EXCLUDE (source_file, fasta_id),
    -- Residue classes (Sneath 1966 / standard groupings), derived from the AAC.
    a.alanine_pct + a.valine_pct + a.isoleucine_pct + a.leucine_pct
        + a.methionine_pct + a.phenylalanine_pct + a.tryptophan_pct
        + a.tyrosine_pct                        AS hydrophobic_pct,
    a.serine_pct + a.threonine_pct + a.asparagine_pct + a.cysteine_pct
        + a.glutamine_pct                       AS polar_pct,
    a.lysine_pct + a.arginine_pct + a.histidine_pct AS positive_pct,
    a.aspartic_acid_pct + a.glutamic_acid_pct   AS negative_pct,
    a.lysine_pct + a.arginine_pct + a.histidine_pct
        + a.aspartic_acid_pct + a.glutamic_acid_pct AS charged_pct,
    a.phenylalanine_pct + a.tryptophan_pct + a.tyrosine_pct AS aromatic_pct,
    a.cysteine_pct + a.methionine_pct           AS sulfur_pct
FROM proteins p
JOIN sequence_features f USING (source_file, fasta_id)
JOIN amino_acid_composition a USING (source_file, fasta_id);

CREATE OR REPLACE VIEW protein_annotation_summary AS
SELECT
    p.source_file,
    p.fasta_id,
    p.uniprot_accession,
    a.protein_name,
    a.organism,
    a.taxon_id,
    a.status AS annotation_status,   -- NULL = never queried, matched, not_found
    a.retrieved_at,
    count(DISTINCT g.go_id) AS go_terms,
    count(DISTINCT g.go_id) FILTER (WHERE g.aspect = 'molecular_function') AS go_mf_terms,
    count(DISTINCT g.go_id) FILTER (WHERE g.aspect = 'biological_process') AS go_bp_terms,
    count(DISTINCT g.go_id) FILTER (WHERE g.aspect = 'cellular_component') AS go_cc_terms
FROM proteins p
LEFT JOIN uniprot_annotations a USING (source_file, fasta_id)
LEFT JOIN protein_go_annotations g USING (source_file, fasta_id)
GROUP BY ALL;
