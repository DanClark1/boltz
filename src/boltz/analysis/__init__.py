"""Interpretability analysis tools for Boltz models."""

from boltz.analysis.interp import (
    register_metadata_hook,
    decode_res_types,
    load_structure_residues,
    compute_ca_coords,
    compute_distance_matrix,
    compute_contact_matrix,
    plot_residue_type_attention,
    plot_top_attended_residues_for_head,
    plot_top_residue_type_heatmap,
    compute_layer_structure_correlations,
    plot_bias_vs_structure,
    plot_structure_correlation_heatmap,
    plot_full_extended_analysis,
    AA_CATEGORIES,
    AA_1LETTER,
    AA_TO_CATEGORY,
)

__all__ = [
    "register_metadata_hook",
    "decode_res_types",
    "load_structure_residues",
    "compute_ca_coords",
    "compute_distance_matrix",
    "compute_contact_matrix",
    "plot_residue_type_attention",
    "plot_top_attended_residues_for_head",
    "plot_top_residue_type_heatmap",
    "compute_layer_structure_correlations",
    "plot_bias_vs_structure",
    "plot_structure_correlation_heatmap",
    "plot_full_extended_analysis",
    "AA_CATEGORIES",
    "AA_1LETTER",
    "AA_TO_CATEGORY",
]
