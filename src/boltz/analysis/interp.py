"""Interpretability analysis for Boltz pairformer attention.

Provides two main analysis extensions:

1. **Residue type analysis** – for both the semantic (QK content) and geometric
   (pairwise bias) attention components, identifies which amino acid types
   receive the most attention and plots the distribution by residue category.

2. **Structure correlation analysis** – compares each attention head's bias
   matrix to the predicted 3D structure (Cα pairwise distance matrix) and
   measures the Spearman correlation, telling you whether "geometric" heads
   are actually attending to structurally proximal residues.

Typical Colab workflow
----------------------
*During prediction (add to your hook-registration cell):*

    from boltz.analysis.interp import register_metadata_hook, decode_res_types
    metadata, meta_handle = register_metadata_hook(model)
    # … run boltz.predict() …
    res_names = decode_res_types(metadata['res_type'], metadata.get('token_pad_mask'))
    torch.save({'res_names': res_names}, '/content/drive/MyDrive/metadata.pt')
    meta_handle.remove()

*During analysis:*

    from boltz.analysis.interp import (
        load_structure_residues, compute_ca_coords, compute_distance_matrix,
        plot_residue_type_attention, plot_bias_vs_structure,
        compute_layer_structure_correlations, plot_structure_correlation_heatmap,
    )
    # Load structure for Cα extraction (alternative to metadata hook)
    res_names, _ = load_structure_residues('prot_no_msa/processed', 'prot_no_msa')
    coords = torch.load('coords.pt')
    ca_coords = compute_ca_coords(coords, res_names)
    ca_dist   = compute_distance_matrix(ca_coords)

    plot_residue_type_attention(activations, layer_names, res_names)
    plot_bias_vs_structure(activations, layer_names, ca_dist, res_names,
                           layer_idx=0, head_idx=0)
    bias_corr    = compute_layer_structure_correlations(activations, layer_names, ca_dist, 'bias')
    content_corr = compute_layer_structure_correlations(activations, layer_names, ca_dist, 'content')
    plot_structure_correlation_heatmap(bias_corr, content_corr, layer_labels)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Amino-acid metadata
# ---------------------------------------------------------------------------

AA_CATEGORIES: dict[str, list[str]] = {
    "Hydrophobic": ["ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"],
    "Polar":       ["SER", "THR", "CYS", "TYR", "ASN", "GLN"],
    "Charged+":    ["LYS", "ARG", "HIS"],
    "Charged-":    ["ASP", "GLU"],
    "Special":     ["GLY"],
}

AA_TO_CATEGORY: dict[str, str] = {
    aa: cat for cat, aas in AA_CATEGORIES.items() for aa in aas
}
AA_TO_CATEGORY["UNK"] = "Unknown"

_CATEGORY_COLORS: dict[str, str] = {
    "Hydrophobic": "#E07B7B",
    "Polar":       "#4ECDC4",
    "Charged+":    "#45B7D1",
    "Charged-":    "#96CEB4",
    "Special":     "#F4D03F",
    "Unknown":     "#BDC3C7",
}

AA_1LETTER: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "UNK": "X",
}

# Ordered standard 20 AAs for consistent axis labelling
_STANDARD_AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# ---------------------------------------------------------------------------
# 1. Metadata capture hook
# ---------------------------------------------------------------------------

def register_metadata_hook(model) -> tuple[dict, object]:
    """Register a pre-hook on the input embedder to capture batch metadata.

    Call **before** ``boltz.predict()``.  The returned *metadata* dict is
    populated during the forward pass with the tensors needed for residue-type
    decoding.

    Parameters
    ----------
    model:
        A loaded Boltz model (from ``boltz.load_model()``).

    Returns
    -------
    metadata : dict
        Populated after ``predict()`` with keys ``'res_type'`` and
        ``'token_pad_mask'``.
    handle : torch.utils.hooks.RemovableHook
        Call ``handle.remove()`` once prediction is done.

    Example
    -------
    ::

        metadata, handle = register_metadata_hook(model)
        results = boltz.predict(model, yaml_path, ...)
        res_names = decode_res_types(metadata['res_type'],
                                     metadata.get('token_pad_mask'))
        handle.remove()
    """
    metadata: dict = {}

    def _pre_hook(module, args):
        # args[0] is the batch/feats dict
        if not args:
            return
        batch = args[0]
        if not isinstance(batch, dict):
            return
        if "res_type" in batch:
            metadata["res_type"] = batch["res_type"].detach().cpu()
        if "token_pad_mask" in batch:
            metadata["token_pad_mask"] = batch["token_pad_mask"].detach().cpu()

    handle = model.input_embedder.register_forward_pre_hook(_pre_hook)
    return metadata, handle


def decode_res_types(
    res_type: torch.Tensor,
    token_mask: Optional[torch.Tensor] = None,
) -> list[str]:
    """Convert the batch ``res_type`` tensor to a list of residue names.

    The featurizer stores ``res_type`` as a one-hot tensor with shape
    ``[batch, num_tokens, num_token_types]``.  This function decodes it.

    Parameters
    ----------
    res_type : torch.Tensor
        The ``feats['res_type']`` tensor (one-hot or integer-indexed).
    token_mask : torch.Tensor, optional
        The ``feats['token_pad_mask']`` tensor.  Only positions where
        ``mask == 1`` are included in the output.

    Returns
    -------
    list[str]
        Residue names such as ``['ALA', 'GLY', 'TRP', ...]``.
    """
    from boltz.data.const import tokens as _token_list

    # Handle batch dimension
    if res_type.dim() == 3:
        res_type = res_type[0]          # [num_tokens, num_classes]
    if res_type.dim() == 2:
        ids = torch.argmax(res_type, dim=-1).numpy()   # one-hot → int
    else:
        ids = res_type.numpy()                          # already integer

    if token_mask is not None:
        if token_mask.dim() == 2:
            token_mask = token_mask[0]
        mask = token_mask.numpy().astype(bool)
        ids = ids[mask]

    return [
        (_token_list[int(i)] if 0 <= int(i) < len(_token_list) else "UNK")
        for i in ids
    ]


# ---------------------------------------------------------------------------
# 2. Structure / coordinate utilities
# ---------------------------------------------------------------------------

def load_structure_residues(
    processed_dir: str | Path,
    input_stem: Optional[str] = None,
) -> tuple[list[str], np.ndarray]:
    """Load residue names from boltz's processed structure NPZ file.

    After ``boltz.predict()`` runs, it saves a processed structure to
    ``<out_dir>/processed/structures/<input_name>.npz``.  This function
    reads that file and returns residue names alongside their absolute
    center-atom (Cα) indices into the full atom array – useful for
    extracting Cα positions from predicted coordinates.

    Parameters
    ----------
    processed_dir : str or Path
        Path to the ``processed`` subdirectory created by ``boltz.predict()``.
        E.g. ``'prot_no_msa/processed'``.
    input_stem : str, optional
        Stem of the input YAML (without extension).  If ``None`` the first
        ``.npz`` in ``structures/`` is used.

    Returns
    -------
    res_names : list[str]
        Residue name for each residue in chain order (e.g. ``'ALA'``).
    atom_center : np.ndarray
        Absolute atom index of the center atom (Cα) for each residue,
        indexed into the featurized atom array that matches the predicted
        coordinate tensor.
    """
    from boltz.data.const import canonical_tokens, tokens as token_list

    processed_dir = Path(processed_dir)
    struct_dir = processed_dir / "structures"

    if input_stem is not None:
        path = struct_dir / f"{input_stem}.npz"
    else:
        paths = sorted(struct_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz files found in {struct_dir}")
        path = paths[0]

    data = np.load(path, allow_pickle=True)
    residues = data["residues"]  # structured array with Residue dtype

    res_names = [str(r["name"]).strip() for r in residues]
    atom_center = np.array([int(r["atom_center"]) for r in residues], dtype=np.int64)

    return res_names, atom_center


def compute_ca_coords(
    coords: torch.Tensor,
    res_names: list[str],
    sample_idx: int = 0,
) -> np.ndarray:
    """Extract Cα (or nucleic-acid C1') coordinates for each residue.

    This function uses the ``ref_atoms`` lookup from ``boltz.data.const``
    to determine the local atom offset of the center atom within each
    residue's atom block, then accumulates atom counts to compute absolute
    indices into the predicted coordinate tensor.

    Works for any mix of protein / nucleic-acid residues (handled via
    ``ref_atoms``).  Tokens with zero atoms (``PAD``, ``'-'``) are skipped
    automatically.

    Parameters
    ----------
    coords : torch.Tensor
        Predicted atom coordinates with shape
        ``[diffusion_samples, num_atoms, 3]`` or ``[num_atoms, 3]``.
    res_names : list[str]
        Ordered residue names (one per valid token).  Obtain via
        :func:`decode_res_types` or :func:`load_structure_residues`.
    sample_idx : int
        Which diffusion sample to use. Default ``0``.

    Returns
    -------
    ca_coords : np.ndarray, shape ``[num_valid_residues, 3]``
        Cα (or C1') coordinates for each residue that has a known atom
        layout.  Residues that are not in ``ref_atoms`` are skipped, so
        the returned array may be shorter than ``res_names``.
    """
    from boltz.data.const import ref_atoms

    if coords.dim() == 3:
        xyz = coords[sample_idx].numpy()
    else:
        xyz = coords.numpy()

    ca_list: list[np.ndarray] = []
    atom_start = 0

    for res_name in res_names:
        atoms_in_res = ref_atoms.get(res_name, [])
        n_atoms = len(atoms_in_res)

        if n_atoms == 0:
            # PAD / gap token – no atoms, skip
            continue

        # Determine center-atom local offset within this residue's block
        if "CA" in atoms_in_res:
            offset = atoms_in_res.index("CA")
        elif "C1'" in atoms_in_res:
            offset = atoms_in_res.index("C1'")
        else:
            offset = 0  # fallback: first atom

        ca_idx = atom_start + offset
        if ca_idx < len(xyz):
            ca_list.append(xyz[ca_idx])
        else:
            ca_list.append(np.zeros(3, dtype=np.float32))

        atom_start += n_atoms

    return np.array(ca_list, dtype=np.float32)


def compute_distance_matrix(ca_coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Cα distance matrix.

    Parameters
    ----------
    ca_coords : np.ndarray, shape ``[N, 3]``

    Returns
    -------
    dist : np.ndarray, shape ``[N, N]``
        Pairwise Euclidean distances in Ångströms.
    """
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def compute_contact_matrix(
    ca_coords: np.ndarray,
    threshold: float = 8.0,
) -> np.ndarray:
    """Binary contact map (``True`` where Cα distance < *threshold* Å).

    Parameters
    ----------
    ca_coords : np.ndarray, shape ``[N, 3]``
    threshold : float
        Default ``8.0`` Å.

    Returns
    -------
    contacts : np.ndarray of bool, shape ``[N, N]``
    """
    return compute_distance_matrix(ca_coords) < threshold


# ---------------------------------------------------------------------------
# 3. Attention helpers (shared by analysis functions)
# ---------------------------------------------------------------------------

def _get_attention_maps(data: dict, component: str) -> tuple[np.ndarray, int]:
    """Recompute softmaxed attention maps for all heads in one layer.

    Parameters
    ----------
    data : dict
        One entry from the ``activations`` dict, with keys ``'q'``, ``'k'``,
        ``'bias'``.
    component : str
        One of ``'bias'``, ``'content'``, or ``'full'``.

    Returns
    -------
    attn : np.ndarray, shape ``[num_heads, N, N]``
    num_heads : int
    """
    q_raw = data["q"].float()
    k_raw = data["k"].float()
    bias = data["bias"].float()

    B, N, Hidden = q_raw.shape

    if bias.shape[-1] == N:
        num_heads = bias.shape[1]   # already [B, H, N, N]
    else:
        num_heads = bias.shape[-1]  # [B, N, N, H] → permute
        bias = bias.permute(0, 3, 1, 2)

    head_dim = Hidden // num_heads
    q = q_raw.view(B, N, num_heads, head_dim).transpose(1, 2)
    k = k_raw.view(B, N, num_heads, head_dim).transpose(1, 2)

    attn_maps = np.empty((num_heads, N, N), dtype=np.float32)
    for h in range(num_heads):
        q_h = q[0, h]
        k_h = k[0, h]
        b_h = bias[0, h]
        content = torch.matmul(q_h, k_h.T) / (head_dim ** 0.5)

        if component == "bias":
            attn_h = torch.softmax(b_h, dim=-1)
        elif component == "content":
            attn_h = torch.softmax(content, dim=-1)
        elif component == "full":
            attn_h = torch.softmax(content + b_h, dim=-1)
        else:
            raise ValueError(f"Unknown component: {component!r}")

        attn_maps[h] = attn_h.numpy()

    return attn_maps, num_heads


def _layer_idx_from_name(name: str) -> int:
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# 4. Residue-type attention analysis
# ---------------------------------------------------------------------------

def _aggregate_type_attention(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    component: str,
) -> dict[str, list[float]]:
    """Return {res_name: [attention_values…]} across all layers/heads."""
    type_attn: dict[str, list[float]] = {}

    for name in layer_names:
        attn_maps, num_heads = _get_attention_maps(activations[name], component)
        N = min(attn_maps.shape[-1], len(res_names))

        for h in range(num_heads):
            # Mean attention *received* by each position (column mean)
            per_pos = attn_maps[h, :N, :N].mean(axis=0)  # shape [N]
            for i, res in enumerate(res_names[:N]):
                type_attn.setdefault(res, []).append(float(per_pos[i]))

    return type_attn


def plot_residue_type_attention(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    fig_title: str = "Residue Type Attention Analysis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar charts showing which amino acid types receive the most attention.

    Two panels side-by-side: semantic (QK content) vs geometric (pairwise
    bias).  Bars are coloured by biochemical category.

    Parameters
    ----------
    activations : dict
        The captured activations dict (keys are layer names).
    layer_names : list[str]
        Ordered list of layer names, as used in the KL analysis.
    res_names : list[str]
        Residue name for each token position.  Obtain from
        :func:`decode_res_types` or :func:`load_structure_residues`.
    fig_title : str
    save_path : str, optional
        If given, saves the figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    content_ta = _aggregate_type_attention(
        activations, layer_names, res_names, "content"
    )
    bias_ta = _aggregate_type_attention(
        activations, layer_names, res_names, "bias"
    )

    # Only protein residues present in the sequence
    present = {r for r in res_names if r in AA_1LETTER}
    # Sort in canonical AA order
    unique_res = [r for r in _STANDARD_AA_ORDER if r in present]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(fig_title, fontsize=14, weight="bold")

    for ax, (label, ta) in zip(
        axes,
        [
            ("Semantic – Content (QK\u1d40)", content_ta),
            ("Geometric – Pairwise Bias", bias_ta),
        ],
    ):
        if not unique_res:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        means = [np.mean(ta[r]) if r in ta else 0.0 for r in unique_res]
        colors = [
            _CATEGORY_COLORS.get(AA_TO_CATEGORY.get(r, "Unknown"), "#BDC3C7")
            for r in unique_res
        ]
        xlabels = [f"{AA_1LETTER.get(r, r)}\n({r})" for r in unique_res]

        # Sort descending by mean attention
        order = np.argsort(means)[::-1]
        means   = [means[i]   for i in order]
        colors  = [colors[i]  for i in order]
        xlabels = [xlabels[i] for i in order]

        ax.bar(range(len(xlabels)), means, color=colors)
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("Mean attention weight (avg over all layers & heads)")
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=c, label=cat)
        for cat, c in _CATEGORY_COLORS.items()
        if cat != "Unknown"
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.06), frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_top_attended_residues_for_head(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    layer_idx: int,
    head_idx: int,
    top_k: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """For a specific layer/head, show which residue *positions* receive the
    most attention for the bias vs content components.

    Bars are coloured by biochemical category and labelled with position
    index and one-letter code.

    Parameters
    ----------
    activations, layer_names, res_names
        As in other functions.
    layer_idx : int
        Index into *layer_names*.
    head_idx : int
        Head index (0-based).
    top_k : int
        Number of top positions to show.  Default 15.
    save_path : str, optional
    """
    name = layer_names[layer_idx]
    lay_real = _layer_idx_from_name(name)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"Top-{top_k} Attended Positions — Layer {lay_real}, Head {head_idx}",
        fontsize=13, weight="bold",
    )

    for ax, component in zip(axes, ["content", "bias"]):
        attn_maps, num_heads = _get_attention_maps(activations[name], component)
        if head_idx >= num_heads:
            ax.text(0.5, 0.5, f"Head {head_idx} not found", ha="center",
                    va="center", transform=ax.transAxes)
            continue

        N = min(attn_maps.shape[-1], len(res_names))
        attn = attn_maps[head_idx, :N, :N]
        per_pos = attn.mean(axis=0)  # [N] – mean attention received

        top_idx = np.argsort(per_pos)[::-1][:top_k]
        vals    = per_pos[top_idx]
        colors  = [
            _CATEGORY_COLORS.get(AA_TO_CATEGORY.get(res_names[i], "Unknown"), "#BDC3C7")
            for i in top_idx
        ]
        xlabels = [
            f"{i}: {AA_1LETTER.get(res_names[i], '?')}\n({res_names[i]})"
            for i in top_idx
        ]

        ax.bar(range(top_k), vals, color=colors)
        ax.set_xticks(range(top_k))
        ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Mean attention received")
        comp_label = "Semantic (QK Content)" if component == "content" else "Geometric (Bias)"
        ax.set_title(comp_label)
        ax.grid(True, alpha=0.3, axis="y")

    legend_handles = [
        mpatches.Patch(facecolor=c, label=cat)
        for cat, c in _CATEGORY_COLORS.items()
        if cat != "Unknown"
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 5. Structure correlation analysis
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats
    if len(x) < 5:
        return 0.0
    r, _ = stats.spearmanr(x, y)
    return float(r) if np.isfinite(r) else 0.0


def compute_layer_structure_correlations(
    activations: dict,
    layer_names: list[str],
    ca_dist_matrix: np.ndarray,
    component: str = "bias",
    seq_sep: int = 3,
) -> np.ndarray:
    """Spearman correlation between attention weights and structural proximity.

    For each layer × head, computes the Spearman r between the softmaxed
    attention values and *1 / (Cα distance + 1)* (i.e. higher = closer in
    3D).  A positive correlation means the head preferentially attends to
    structurally nearby residues.

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    ca_dist_matrix : np.ndarray, shape ``[N, N]``
        Pairwise Cα distance matrix from :func:`compute_distance_matrix`.
    component : {'bias', 'content', 'full'}
    seq_sep : int
        Exclude residue pairs closer than this in sequence (to avoid trivial
        local attention dominating the correlation).  Default ``3``.

    Returns
    -------
    corr : np.ndarray, shape ``[num_layers, num_heads]``
        Spearman r values.
    """
    num_layers = len(layer_names)
    corr_matrix: Optional[np.ndarray] = None

    for i, name in enumerate(layer_names):
        attn_maps, num_heads = _get_attention_maps(activations[name], component)

        if corr_matrix is None:
            corr_matrix = np.zeros((num_layers, num_heads), dtype=np.float32)

        N = min(attn_maps.shape[-1], ca_dist_matrix.shape[0])

        # Build a flat off-diagonal mask that excludes nearby-in-sequence pairs
        row_idx, col_idx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        mask = (np.abs(row_idx - col_idx) >= seq_sep)       # exclude nearby in seq
        mask &= (ca_dist_matrix[:N, :N] > 0)               # exclude zero-coord atoms
        mask &= ~np.eye(N, dtype=bool)                      # exclude diagonal

        inv_dist = 1.0 / (ca_dist_matrix[:N, :N] + 1.0)    # structural proximity

        for h in range(num_heads):
            attn_flat = attn_maps[h, :N, :N][mask]
            prox_flat = inv_dist[mask]
            corr_matrix[i, h] = _spearman(attn_flat, prox_flat)

    return corr_matrix if corr_matrix is not None else np.zeros((num_layers, 1))


def plot_bias_vs_structure(
    activations: dict,
    layer_names: list[str],
    ca_dist_matrix: np.ndarray,
    res_names: list[str],
    layer_idx: int,
    head_idx: int,
    zoom: int = 80,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side comparison of bias / content attention vs predicted structure.

    Shows six panels:
    - Bias attention heatmap
    - Content attention heatmap
    - Full (bias + content) attention heatmap
    - Structural proximity (1/(d+1)) heatmap
    - Contact map (< 8 Å)
    - Scatter: bias attention vs Cα distance (with Pearson r)

    Parameters
    ----------
    activations, layer_names, res_names
        Standard analysis inputs.
    ca_dist_matrix : np.ndarray, shape ``[N, N]``
    layer_idx : int
        Index into *layer_names*.
    head_idx : int
    zoom : int
        Crop all heatmaps to the first *zoom* residues for legibility.
    save_path : str, optional
    """
    name = layer_names[layer_idx]
    lay_real = _layer_idx_from_name(name)

    bias_maps, num_heads = _get_attention_maps(activations[name], "bias")
    cont_maps, _         = _get_attention_maps(activations[name], "content")
    full_maps, _         = _get_attention_maps(activations[name], "full")

    if head_idx >= num_heads:
        raise ValueError(f"head_idx={head_idx} but only {num_heads} heads in this layer.")

    N_full = bias_maps.shape[-1]
    N = min(N_full, ca_dist_matrix.shape[0], len(res_names), zoom)

    attn_bias = bias_maps[head_idx, :N, :N]
    attn_cont = cont_maps[head_idx, :N, :N]
    attn_full = full_maps[head_idx, :N, :N]

    dist_crop = ca_dist_matrix[:N, :N]
    inv_dist  = 1.0 / (dist_crop + 1.0)
    inv_dist_n = inv_dist / (inv_dist.max() + 1e-9)
    contacts  = (dist_crop < 8.0).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Layer {lay_real}, Head {head_idx}: Attention vs Predicted Structure",
        fontsize=14, weight="bold",
    )

    # Row 0: attention matrices
    sns.heatmap(attn_bias, cmap="cividis", ax=axes[0, 0], cbar=True,
                vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    axes[0, 0].set_title("Bias Attention (Geometric)")

    sns.heatmap(attn_cont, cmap="magma", ax=axes[0, 1], cbar=True,
                vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    axes[0, 1].set_title("Content Attention (Semantic)")

    sns.heatmap(attn_full, cmap="viridis", ax=axes[0, 2], cbar=True,
                vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    axes[0, 2].set_title("Full Attention (Bias + Content)")

    # Row 1: structure
    sns.heatmap(inv_dist_n, cmap="hot_r", ax=axes[1, 0], cbar=True,
                vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    axes[1, 0].set_title("Structural Proximity  1/(Cα dist + 1)")

    sns.heatmap(contacts, cmap="Greens", ax=axes[1, 1], cbar=True,
                vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    axes[1, 1].set_title("Contact Map (< 8 Å)")

    # Scatter: bias attention vs distance
    ax = axes[1, 2]
    mask_s = ~np.eye(N, dtype=bool)
    d_flat = dist_crop[mask_s]
    a_flat = attn_bias[mask_s]

    # Subsample for speed if very large
    if len(d_flat) > 20_000:
        idx = np.random.choice(len(d_flat), 20_000, replace=False)
        d_flat, a_flat = d_flat[idx], a_flat[idx]

    ax.scatter(d_flat, a_flat, alpha=0.06, s=4, c="#7B2FBE", rasterized=True)

    # Trend / Pearson r
    valid = (d_flat > 0) & (d_flat < 60)
    if valid.sum() > 20:
        from scipy import stats as _stats
        slope, intercept, r_val, _, _ = _stats.linregress(d_flat[valid], a_flat[valid])
        x_line = np.linspace(d_flat[valid].min(), d_flat[valid].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
                label=f"Pearson r = {r_val:.3f}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Cα distance (Å)")
    ax.set_ylabel("Bias attention weight")
    ax.set_title("Bias Attention vs Cα Distance")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_structure_correlation_heatmap(
    corr_bias: np.ndarray,
    corr_content: np.ndarray,
    layer_labels: list[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of per-head structure correlation (Spearman r) for bias and content.

    Red = positive correlation (head attends to structurally close residues),
    Blue = negative correlation.

    Parameters
    ----------
    corr_bias : np.ndarray, shape ``[num_layers, num_heads]``
        From :func:`compute_layer_structure_correlations` with
        ``component='bias'``.
    corr_content : np.ndarray, shape ``[num_layers, num_heads]``
    layer_labels : list[int]
        Layer depth labels (e.g. ``[0, 1, 2, …]``).
    save_path : str, optional
    """
    num_heads = corr_bias.shape[1]
    vmax = max(np.abs(corr_bias).max(), np.abs(corr_content).max(), 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Attention–Structure Correlation (Spearman r with 1/Cα-distance)",
        fontsize=13, weight="bold",
    )

    for ax, data, title in zip(
        axes,
        [corr_bias, corr_content],
        ["Bias (Geometric) ↔ Structure", "Content (Semantic) ↔ Structure"],
    ):
        sns.heatmap(
            data,
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            yticklabels=layer_labels,
            xticklabels=[f"H{i}" for i in range(num_heads)],
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer depth")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 6. Omnibus extended analysis
# ---------------------------------------------------------------------------

def plot_full_extended_analysis(
    activations: dict,
    layer_names: list[str],
    ca_dist_matrix: np.ndarray,
    res_names: list[str],
    geo_scores: np.ndarray,
    sem_scores: np.ndarray,
    layer_labels: list[int],
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, np.ndarray, np.ndarray]:
    """Six-panel summary combining KL scores with structure correlation.

    Panels:
    - (0,0) Geometric KL heatmap  (0,1) Semantic KL heatmap
    - (1,0) Bias×Structure corr   (1,1) Content×Structure corr
    - (2,0) Mean-per-layer line   (2,1) KL score vs structure corr scatter

    Parameters
    ----------
    activations, layer_names, res_names
        Standard inputs.
    ca_dist_matrix : np.ndarray, shape ``[N, N]``
    geo_scores, sem_scores : np.ndarray, shape ``[num_layers, num_heads]``
        Normalised KL scores from the original permutation analysis.
    layer_labels : list[int]
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    bias_corr : np.ndarray
    content_corr : np.ndarray
    """
    num_heads = geo_scores.shape[1]

    print("Computing bias–structure correlations …")
    bias_corr = compute_layer_structure_correlations(
        activations, layer_names, ca_dist_matrix, "bias"
    )
    print("Computing content–structure correlations …")
    content_corr = compute_layer_structure_correlations(
        activations, layer_names, ca_dist_matrix, "content"
    )

    vmax_corr = max(np.abs(bias_corr).max(), np.abs(content_corr).max(), 0.05)

    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Extended Pairformer Interpretability Analysis",
                 fontsize=14, weight="bold")

    # Row 0: KL importance
    sns.heatmap(geo_scores, cmap="Reds", vmin=0, vmax=1,
                yticklabels=layer_labels,
                xticklabels=[f"H{i}" for i in range(num_heads)],
                ax=axes[0, 0])
    axes[0, 0].set_title("Geometric Importance (KL, Bias permuted)")
    axes[0, 0].set_xlabel("Head"); axes[0, 0].set_ylabel("Layer")

    sns.heatmap(sem_scores, cmap="Blues", vmin=0, vmax=1,
                yticklabels=layer_labels,
                xticklabels=[f"H{i}" for i in range(num_heads)],
                ax=axes[0, 1])
    axes[0, 1].set_title("Semantic Importance (KL, Content permuted)")
    axes[0, 1].set_xlabel("Head"); axes[0, 1].set_ylabel("Layer")

    # Row 1: structure correlations
    sns.heatmap(bias_corr, cmap="RdBu_r", center=0,
                vmin=-vmax_corr, vmax=vmax_corr,
                yticklabels=layer_labels,
                xticklabels=[f"H{i}" for i in range(num_heads)],
                ax=axes[1, 0])
    axes[1, 0].set_title("Bias × Structural Proximity (Spearman r)")
    axes[1, 0].set_xlabel("Head"); axes[1, 0].set_ylabel("Layer")

    sns.heatmap(content_corr, cmap="RdBu_r", center=0,
                vmin=-vmax_corr, vmax=vmax_corr,
                yticklabels=layer_labels,
                xticklabels=[f"H{i}" for i in range(num_heads)],
                ax=axes[1, 1])
    axes[1, 1].set_title("Content × Structural Proximity (Spearman r)")
    axes[1, 1].set_xlabel("Head"); axes[1, 1].set_ylabel("Layer")

    # Row 2: mean-per-layer line plot
    ax = axes[2, 0]
    ax.plot(layer_labels, geo_scores.mean(axis=1),   "r-o",  ms=4, label="Geo KL (bias)")
    ax.plot(layer_labels, sem_scores.mean(axis=1),   "b-s",  ms=4, label="Sem KL (content)")
    ax.plot(layer_labels, bias_corr.mean(axis=1),    "r--^", ms=4, label="Bias↔Structure r")
    ax.plot(layer_labels, content_corr.mean(axis=1), "b--v", ms=4, label="Content↔Structure r")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Layer depth"); ax.set_ylabel("Score / Correlation")
    ax.set_title("Layer-wise Summary")
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)

    # Row 2 right: KL vs structure-corr scatter
    ax = axes[2, 1]
    ax.scatter(geo_scores.flatten(), bias_corr.flatten(),
               c="red",  alpha=0.4, s=15, label="Bias (geo)")
    ax.scatter(sem_scores.flatten(), content_corr.flatten(),
               c="blue", alpha=0.4, s=15, label="Content (sem)")
    ax.set_xlabel("KL Importance Score (normalised)")
    ax.set_ylabel("Structure Correlation (Spearman r)")
    ax.set_title("KL Importance vs Structure Correlation\n(per head)")
    ax.axhline(0, color="gray", ls=":", alpha=0.3)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, bias_corr, content_corr


# ---------------------------------------------------------------------------
# 7. Top residue-type heatmap (per layer × head)
# ---------------------------------------------------------------------------

def plot_top_residue_type_heatmap(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    layer_labels: list[int],
    sem_scores: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    annotate: Optional[bool] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap showing *which amino acid type* is most attended to per head.

    Two panels side-by-side:

    - **Left** – top attended residue type from the geometric (pairwise bias)
      component.
    - **Right** – top attended residue type from the semantic (QK content)
      component.

    For each layer × head the cell shows the amino acid type (e.g. LEU, GLY)
    whose positions receive the highest *mean* attention in that head's
    softmaxed attention map.  Cells are coloured by amino acid identity using
    a 20-colour qualitative palette; a shared colorbar labels each colour.
    One-letter codes are printed inside the cells when legible.

    Heads with a normalised semantic KL score below *threshold* are blacked
    out on both panels, because their residue routing is too uniform to be
    meaningful.

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    res_names : list[str]
        Residue names for each token position.
    layer_labels : list[int]
        Layer depth labels for the y-axis.
    sem_scores : np.ndarray, optional
        Normalised semantic KL scores with shape ``[num_layers, num_heads]``
        (i.e. the ``sem_scores`` array from the permutation analysis).
        Heads where ``sem_scores[layer, head] < threshold`` are blacked out.
    threshold : float
        Minimum semantic score for a head to be shown.  Default ``0.3``.
    annotate : bool, optional
        Whether to print the 1-letter code inside each cell.  Defaults to
        ``True`` when ``num_layers * num_heads <= 400``, else ``False``.
    save_path : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable

    # Amino acid encoding – 20 standard AAs + UNK
    aa_order = _STANDARD_AA_ORDER + ["UNK"]
    aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
    n_colors = len(aa_order)

    # Use a 21-colour qualitative palette (tab20 + one extra for UNK)
    base_cmap = plt.cm.get_cmap("tab20", 20)
    colors = [base_cmap(i) for i in range(20)] + [(0.7, 0.7, 0.7, 1.0)]  # grey for UNK
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors, name="aa_cmap")

    num_layers = len(layer_names)
    num_heads: Optional[int] = None
    bias_matrix: Optional[np.ndarray] = None
    cont_matrix: Optional[np.ndarray] = None

    for i, name in enumerate(layer_names):
        attn_bias, nh = _get_attention_maps(activations[name], "bias")
        attn_cont, _  = _get_attention_maps(activations[name], "content")

        if num_heads is None:
            num_heads = nh
            bias_matrix = np.full((num_layers, num_heads), n_colors - 1, dtype=int)
            cont_matrix = np.full((num_layers, num_heads), n_colors - 1, dtype=int)

        N = min(attn_bias.shape[-1], len(res_names))

        for h in range(num_heads):
            # Mean attention *received* by each position (column mean)
            bias_per_pos = attn_bias[h, :N, :N].mean(axis=0)
            cont_per_pos = attn_cont[h, :N, :N].mean(axis=0)

            # Accumulate by residue type
            type_bias: dict[str, list[float]] = {}
            type_cont: dict[str, list[float]] = {}
            for j, res in enumerate(res_names[:N]):
                if res in aa_to_idx:
                    type_bias.setdefault(res, []).append(float(bias_per_pos[j]))
                    type_cont.setdefault(res, []).append(float(cont_per_pos[j]))

            if type_bias:
                top_b = max(type_bias, key=lambda r: np.mean(type_bias[r]))
                bias_matrix[i, h] = aa_to_idx.get(top_b, n_colors - 1)
            if type_cont:
                top_c = max(type_cont, key=lambda r: np.mean(type_cont[r]))
                cont_matrix[i, h] = aa_to_idx.get(top_c, n_colors - 1)

    if num_heads is None:
        raise ValueError("No layers processed – check layer_names and activations.")

    # Build the boolean mask (True = black out this cell)
    if sem_scores is not None:
        blackout = sem_scores < threshold          # shape [num_layers, num_heads]
    else:
        blackout = np.zeros((num_layers, num_heads), dtype=bool)

    # Auto-decide annotation
    if annotate is None:
        annotate = (num_layers * num_heads) <= 400

    bounds = np.arange(-0.5, n_colors + 0.5)
    norm = BoundaryNorm(bounds, cmap.N)
    cmap.set_bad("black")   # masked cells → black

    fig_h = max(6, num_layers * 0.35 + 2)
    fig, axes = plt.subplots(1, 2, figsize=(18, fig_h))
    thresh_str = f"  (sem score < {threshold} blacked out)" if sem_scores is not None else ""
    fig.suptitle(f"Top Attended Residue Type per Head{thresh_str}",
                 fontsize=13, weight="bold")

    for ax, matrix, title in zip(
        axes,
        [bias_matrix, cont_matrix],
        ["Geometric (Pairwise Bias)", "Semantic (QK Content)"],
    ):
        # Apply mask: replace blacked-out cells with np.ma.masked
        display = np.ma.masked_where(blackout, matrix.astype(float))

        ax.imshow(
            display,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=7)
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels(layer_labels, fontsize=7)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer depth")
        ax.set_title(title)

        if annotate:
            fontsize = max(4, min(8, int(180 / max(num_layers, num_heads))))
            for li in range(num_layers):
                for hi in range(num_heads):
                    if blackout[li, hi]:
                        continue   # leave blacked-out cells empty
                    val = int(matrix[li, hi])
                    letter = AA_1LETTER.get(aa_order[val], "?")
                    bg = colors[val]
                    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
                    txt_color = "black" if luminance > 0.45 else "white"
                    ax.text(hi, li, letter, ha="center", va="center",
                            fontsize=fontsize, color=txt_color, fontweight="bold")

    # Shared colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cbar.set_ticks(range(n_colors))
    cbar.set_ticklabels(
        [f"{AA_1LETTER.get(aa, aa)}  {aa}" for aa in aa_order],
        fontsize=8,
    )
    cbar.set_label("Top attended residue type", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# 8. Per-layer attention matrix export
# ---------------------------------------------------------------------------

def export_layer_attention_matrices(
    activations: dict,
    layer_names: list[str],
    out_dir: str,
    zoom: int = 80,
    dpi: int = 120,
    zip_output: bool = True,
) -> None:
    """Save one figure per layer showing full / bias / content attention for every head.

    Layout: 3 rows (Full | Bias | Content) × num_heads columns.  Each cell
    is a heatmap of the softmaxed attention matrix cropped to the first *zoom*
    residues.

    Parameters
    ----------
    activations : dict
        Captured activations dict, or the output of :func:`get_recycling_step`.
    layer_names : list[str]
        Ordered layer names (e.g. from ``layer_names.sort(key=get_layer_idx)``).
    out_dir : str
        Folder to write the per-layer PNG files into (created if absent).
    zoom : int
        Crop heatmaps to the first *zoom* residues for legibility.  Default 80.
    dpi : int
        Figure DPI.  Default 120.
    zip_output : bool
        If ``True`` (default), also create ``<out_dir>.zip`` for easy download.
    """
    import shutil

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    components = [
        ("full",    "Full Attention\n(Bias + Content)", "viridis"),
        ("bias",    "Bias Attention\n(Geometric)",      "cividis"),
        ("content", "Content Attention\n(Semantic QK)", "magma"),
    ]

    for name in layer_names:
        layer_idx = _layer_idx_from_name(name)

        maps = {}
        num_heads = None
        for comp, _, _ in components:
            m, nh = _get_attention_maps(activations[name], comp)
            maps[comp] = m
            num_heads = nh

        N = min(maps["full"].shape[-1], zoom)

        # Layout: 3 rows (components) × num_heads columns
        fig_w = max(12, 2.2 * num_heads)
        fig, axes = plt.subplots(3, num_heads, figsize=(fig_w, 9),
                                 gridspec_kw={"hspace": 0.08, "wspace": 0.04})

        fig.suptitle(f"Layer {layer_idx} — Attention Matrices  (zoom={zoom})",
                     fontsize=13, weight="bold", y=1.01)

        for row, (comp, row_label, cmap_name) in enumerate(components):
            for h in range(num_heads):
                ax = axes[row, h]
                data = maps[comp][h, :N, :N]
                ax.imshow(data, cmap=cmap_name, vmin=0, vmax=1,
                          aspect="auto", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])

                if row == 0:
                    ax.set_title(f"H{h}", fontsize=8)
                if h == 0:
                    ax.set_ylabel(row_label, fontsize=8, labelpad=4)

        file_path = out_path / f"Layer_{layer_idx:02d}_attention.png"
        plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved Layer {layer_idx:2d} → {file_path.name}")

    if zip_output:
        zip_path = str(out_path.parent / out_path.name)
        shutil.make_archive(zip_path, "zip", out_path)
        print(f"\nZipped → {zip_path}.zip")


# ---------------------------------------------------------------------------
# 9. Recycling-step utilities
# ---------------------------------------------------------------------------

def get_recycling_step(
    activations: dict,
    step: int = -1,
) -> dict:
    """Extract activations for one recycling step from a multi-step capture.

    When hooks are set up with the recycling-aware pattern (list-appending),
    each ``activations[layer][component]`` is a list of tensors, one per
    recycling step.  This helper returns a standard single-step dict compatible
    with all other analysis functions.

    Parameters
    ----------
    activations : dict
        Multi-step activations captured with the list-appending hook pattern.
    step : int
        Which step to extract.  ``-1`` (default) = final step,  ``0`` = first
        step (before any recycling), etc.

    Returns
    -------
    dict
        Single-step activations in the standard ``{layer: {component: tensor}}``
        format expected by all other functions in this module.
    """
    out: dict = {}
    for layer_name, components in activations.items():
        out[layer_name] = {}
        for comp_name, tensor_list in components.items():
            if isinstance(tensor_list, list):
                # Clamp to the last available step — confidence module layers
                # are only called once (not in the recycling loop) so their
                # lists are shorter than the main pairformer layers.
                n = len(tensor_list)
                actual = step if step >= 0 else n + step
                actual = max(0, min(actual, n - 1))
                out[layer_name][comp_name] = tensor_list[actual]
            else:
                out[layer_name][comp_name] = tensor_list
    return out


def stack_recycling_steps(activations: dict) -> dict:
    """Stack all recycling steps into a single tensor along a new leading dim.

    Returns ``{layer: {component: tensor}}`` where each tensor has shape
    ``[num_steps, ...]`` so you can index/slice across steps.

    Parameters
    ----------
    activations : dict
        Multi-step activations captured with the list-appending hook pattern.

    Returns
    -------
    dict
        ``{layer: {component: stacked_tensor}}``
    """
    out: dict = {}
    for layer_name, components in activations.items():
        out[layer_name] = {}
        for comp_name, tensor_list in components.items():
            if isinstance(tensor_list, list):
                out[layer_name][comp_name] = torch.stack(tensor_list, dim=0)
            else:
                out[layer_name][comp_name] = tensor_list.unsqueeze(0)
    return out


# ---------------------------------------------------------------------------
# 10. KL permutation analysis
# ---------------------------------------------------------------------------

def run_kl_analysis(
    activations: dict,
    layer_names: list[str],
    num_trials: int = 5,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """KL permutation analysis to separate geometric vs semantic attention importance.

    For each layer × head computes:

    - **geo_score**: mean KL(A ‖ A_perm_bias) over random bias permutations.
      Measures how much the pairwise bias (geometric routing) determines attention.
    - **sem_score**: mean KL(A ‖ A_perm_content) over random content permutations.
      Measures how much the QK content (semantic routing) determines attention.

    Parameters
    ----------
    activations : dict
        Single-step activations ``{layer: {'q': tensor, 'k': tensor, 'bias': tensor}}``.
    layer_names : list[str]
        Ordered main-pairformer layer names (filtered to ``pairformer_module.*``).
    num_trials : int
        Number of random permutation trials per head.  Default 5.
    eps : float
        Small constant for log stability.

    Returns
    -------
    geo_raw, sem_raw : np.ndarray, shape ``[num_layers, num_heads]``
        Raw (unnormalised) KL scores.  Normalise by the 95th percentile before
        comparing across structures or recycling steps.
    """
    geo_raw: Optional[np.ndarray] = None
    sem_raw: Optional[np.ndarray] = None

    np.random.seed(42)
    torch.manual_seed(42)

    for i, name in enumerate(layer_names):
        data   = activations[name]
        q_raw  = data["q"].float()
        k_raw  = data["k"].float()
        bias   = data["bias"].float()

        B, N, Hidden = q_raw.shape

        # Normalise bias shape to [B, H, N, N]
        if bias.shape[-1] == N:
            num_heads = bias.shape[1]
        else:
            num_heads = bias.shape[-1]
            bias = bias.permute(0, 3, 1, 2)

        head_dim = Hidden // num_heads
        q = q_raw.view(B, N, num_heads, head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k_raw.view(B, N, num_heads, head_dim).transpose(1, 2)

        if geo_raw is None:
            geo_raw = np.zeros((len(layer_names), num_heads), dtype=np.float32)
            sem_raw = np.zeros((len(layer_names), num_heads), dtype=np.float32)

        for h in range(num_heads):
            q_h = q[0, h]       # [N, D]
            k_h = k[0, h]       # [N, D]
            b_h = bias[0, h]    # [N, N]

            content = torch.matmul(q_h, k_h.T) / (head_dim ** 0.5)  # [N, N]
            A       = torch.softmax(content + b_h, dim=-1)           # [N, N]
            A_safe  = torch.clamp(A, min=eps)

            geo_kl = 0.0
            sem_kl = 0.0
            for _ in range(num_trials):
                # Per-row independent key-dimension shuffle — each query gets its
                # own random ordering of keys.  Using separate perm tensors for
                # geo and sem keeps the two ablations independent.
                perm_geo = torch.rand(N, N).argsort(dim=-1)   # [N, N]
                perm_sem = torch.rand(N, N).argsort(dim=-1)   # [N, N]

                # Geo: scramble key dimension of bias per row → destroys spatial routing
                b_perm = torch.gather(b_h, -1, perm_geo)
                A_geo  = torch.clamp(torch.softmax(content + b_perm, dim=-1), min=eps)
                geo_kl += (A_safe * (A_safe.log() - A_geo.log())).sum(-1).mean().item()

                # Sem: scramble key dimension of content per row → destroys semantic routing
                c_perm = torch.gather(content, -1, perm_sem)
                A_sem  = torch.clamp(torch.softmax(c_perm + b_h, dim=-1), min=eps)
                sem_kl += (A_safe * (A_safe.log() - A_sem.log())).sum(-1).mean().item()

            geo_raw[i, h] = geo_kl / num_trials
            sem_raw[i, h] = sem_kl / num_trials

    return geo_raw, sem_raw


# ---------------------------------------------------------------------------
# 11. Academic publication plots  (RQ1 & RQ2)
# ---------------------------------------------------------------------------

_ACADEMIC_RC: dict = {
    "font.family":           "sans-serif",
    "font.size":             11,
    "axes.labelsize":        12,
    "axes.titlesize":        13,
    "axes.titlepad":         10,
    "axes.labelpad":         6,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.labelsize":       10,
    "ytick.labelsize":       10,
    "legend.fontsize":       10,
    "legend.framealpha":     0.8,
    "figure.titlesize":      14,
    "figure.titleweight":    "bold",
}


def _kl_heatmap_ax(
    ax: "plt.Axes",
    scores: np.ndarray,
    layer_labels: list[int],
    title: str,
    cmap: str,
) -> "plt.cm.ScalarMappable":
    """Draw a single layer×head KL-score heatmap on *ax*.  Returns the image."""
    num_layers, num_heads = scores.shape
    im = ax.imshow(scores, cmap=cmap, vmin=0, vmax=1,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(num_heads))
    ax.set_xticklabels([str(h) for h in range(num_heads)], fontsize=9)
    ax.set_xlabel("Attention head", labelpad=4)

    step = max(1, num_layers // 12)
    ypos = list(range(0, num_layers, step))
    ax.set_yticks(ypos)
    ax.set_yticklabels([str(layer_labels[p]) for p in ypos])
    ax.set_ylabel("Layer", labelpad=4)
    ax.set_title(title, pad=8)
    return im


def plot_kl_heatmaps_separate(
    geo_scores: np.ndarray,
    sem_scores: np.ndarray,
    layer_labels: list[int],
    save_geo: Optional[str] = None,
    save_sem: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    title_geo: Optional[str] = None,
    title_sem: Optional[str] = None,
) -> tuple["plt.Figure", "plt.Figure"]:
    """Two separate academic heatmaps of normalised geo and sem KL scores.

    Parameters
    ----------
    geo_scores, sem_scores : np.ndarray, shape ``[num_layers, num_heads]``
        Normalised (0–1) KL scores from :func:`run_kl_analysis`.
    layer_labels : list[int]
    save_geo, save_sem : str, optional
    figsize : (width, height) in inches, optional
        Override the auto-computed figure size for both panels.
    title_geo : str, optional
        Override the default title for the geometric heatmap.
    title_sem : str, optional
        Override the default title for the semantic heatmap.

    Returns
    -------
    fig_geo, fig_sem : matplotlib.figure.Figure
    geo_scores, sem_scores : np.ndarray
        The arrays that were plotted (pass-through for easy collection across
        multiple proteins).
    """
    num_layers, num_heads = geo_scores.shape
    figs = []

    _title_geo = title_geo if title_geo is not None else "Geometric attention importance\n(pairwise-bias KL score)"
    _title_sem = title_sem if title_sem is not None else "Semantic attention importance\n(content KL score)"
    configs = [
        (geo_scores, _title_geo, "viridis", save_geo),
        (sem_scores, _title_sem, "magma",   save_sem),
    ]

    with plt.rc_context(_ACADEMIC_RC):
        for scores, title, cmap, save_path in configs:
            if figsize is not None:
                fs = figsize
            else:
                fs = (max(5.5, num_heads * 0.5 + 1.5),
                      max(4.5, num_layers * 0.17 + 1.5))
            fig, ax = plt.subplots(figsize=fs)
            im = _kl_heatmap_ax(ax, scores, layer_labels, title, cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Normalised KL score", fontsize=10)
            cbar.ax.tick_params(labelsize=9)
            plt.tight_layout(pad=1.5)
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()
            figs.append(fig)

    return (*figs, geo_scores, sem_scores)


def plot_kl_heatmap_combined(
    geo_scores: np.ndarray,
    sem_scores: np.ndarray,
    layer_labels: list[int],
    save_path: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    title_geo: Optional[str] = None,
    title_sem: Optional[str] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side geo | sem KL heatmap (academic style).

    Parameters
    ----------
    geo_scores, sem_scores : np.ndarray, shape ``[num_layers, num_heads]``
    layer_labels : list[int]
    save_path : str, optional
    figsize : (width, height) in inches, optional
        Override the auto-computed figure size.
    title_geo : str, optional
        Override the default title for the geometric panel.
    title_sem : str, optional
        Override the default title for the semantic panel.
    suptitle : str, optional
        If provided, add a figure-level suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
    geo_scores, sem_scores : np.ndarray
        The arrays that were plotted (pass-through for easy collection across
        multiple proteins).
    """
    num_layers, num_heads = geo_scores.shape

    _title_geo = title_geo if title_geo is not None else "Geometric (bias KL)"
    _title_sem = title_sem if title_sem is not None else "Semantic (content KL)"

    with plt.rc_context(_ACADEMIC_RC):
        if figsize is not None:
            fs = figsize
        else:
            fs = (max(11, num_heads * 1.0 + 3),
                  max(4.5, num_layers * 0.17 + 1.5))
        fig, axes = plt.subplots(1, 2, figsize=fs,
                                 gridspec_kw={"wspace": 0.3})

        for ax, scores, title, cmap in zip(
            axes,
            [geo_scores, sem_scores],
            [_title_geo, _title_sem],
            ["viridis", "magma"],
        ):
            im = _kl_heatmap_ax(ax, scores, layer_labels, title, cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("KL score (norm.)", fontsize=9)

        if suptitle is not None:
            fig.suptitle(suptitle, fontsize=14, weight="bold")
        plt.tight_layout(pad=1.5)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig, geo_scores, sem_scores


def plot_semantic_peaks(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    sem_scores: np.ndarray,
    layer_labels: list[int],
    save_path_line: Optional[str] = None,
    save_path_bars: Optional[str] = None,
    n_peaks: int = 3,
    sem_threshold: float = 0.4,
    figsize_line: Optional[tuple[float, float]] = None,
    figsize_bars: Optional[tuple[float, float]] = None,
    title_line: Optional[str] = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Identify semantic-attention peaks and show which residue categories they target.

    Produces two independent figures:

    - **Line figure**: mean semantic KL score per layer.
    - **Bars figure**: for each peak, a bar chart of mean attention received by
      each amino-acid category (computed from high-sem heads in that layer).

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    res_names : list[str]
    sem_scores : np.ndarray, shape ``[num_layers, num_heads]``
    layer_labels : list[int]
    save_path_line : str, optional
    save_path_bars : str, optional
    n_peaks : int
        Number of peaks to detect and show.  Default 3.
    sem_threshold : float
        Min normalised sem score for a head to be classified as "high-semantic".
    figsize_line : (width, height) in inches, optional
        Override the default line-plot figure size.
    figsize_bars : (width, height) in inches, optional
        Override the default bar-charts figure size.
    title_line : str, optional
        Override the default title for the line-plot figure.

    Returns
    -------
    fig_line, fig_bars : matplotlib.figure.Figure
    layer_mean : np.ndarray, shape ``[num_layers]``
        Per-layer mean semantic score — the line that was plotted.
        Collect this across proteins to overlay on a single figure.
    peak_idxs : list[int]
        Layer indices identified as peaks (into ``layer_names``/``layer_labels``).
    """
    num_layers = len(layer_names)
    layer_mean = sem_scores.mean(axis=1)   # [L]

    # --- Detect peaks ---
    try:
        from scipy.signal import find_peaks as _fp
        peak_idxs, _ = _fp(layer_mean, distance=max(1, num_layers // 10))
        if len(peak_idxs) >= n_peaks:
            peak_idxs = peak_idxs[np.argsort(layer_mean[peak_idxs])[::-1][:n_peaks]]
        else:
            peak_idxs = np.argsort(layer_mean)[::-1][:n_peaks]
    except ImportError:
        peak_idxs = np.argsort(layer_mean)[::-1][:n_peaks]

    peak_idxs = sorted(int(p) for p in peak_idxs)
    peak_colors = ["#E07B7B", "#5BA85B", "#7B7BE0"][:n_peaks]

    with plt.rc_context(_ACADEMIC_RC):

        # ── Figure 1: line plot ──────────────────────────────────────────
        fs_line = figsize_line if figsize_line is not None else (8, 3.5)
        fig_line, ax_top = plt.subplots(figsize=fs_line)

        x = np.arange(num_layers)
        ax_top.plot(x, layer_mean, lw=1.8, color="#5B8DB8", zorder=3,
                    label="Mean sem score")
        ax_top.fill_between(x, 0, layer_mean, alpha=0.12, color="#5B8DB8")

        step = max(1, num_layers // 10)
        ax_top.set_xticks(range(0, num_layers, step))
        ax_top.set_xticklabels([str(layer_labels[p]) for p in range(0, num_layers, step)])
        ax_top.set_xlim(-0.5, num_layers - 0.5)
        ax_top.set_ylim(bottom=0)
        ax_top.set_xlabel("Layer")
        ax_top.set_ylabel("Mean sem KL score")
        _title = title_line if title_line is not None else (
            "Semantic attention importance across layers\n"
            "(peaks = layers with highest content-driven routing)"
        )
        ax_top.set_title(_title)
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.2)
        plt.tight_layout(pad=1.5)
        if save_path_line:
            fig_line.savefig(save_path_line, dpi=150, bbox_inches="tight")
        plt.show()

        # ── Figure 2: residue-category bar charts ────────────────────────
        cats    = list(AA_CATEGORIES.keys()) + ["Unknown"]
        colors  = [_CATEGORY_COLORS[c] for c in cats]
        xlabels = [c.replace("+", "+\n").replace("-", "-\n") for c in cats]

        fs_bars = figsize_bars if figsize_bars is not None else (4.5 * n_peaks, 4)
        fig_bars, axes_bars = plt.subplots(
            1, n_peaks, figsize=fs_bars,
            gridspec_kw={"wspace": 0.4},
        )
        if n_peaks == 1:
            axes_bars = [axes_bars]

        for panel_i, (pidx, pc) in enumerate(zip(peak_idxs, peak_colors)):
            ax = axes_bars[panel_i]

            layer_sem  = sem_scores[pidx]
            high_heads = list(np.where(layer_sem >= sem_threshold)[0])
            if not high_heads:
                high_heads = [int(np.argmax(layer_sem))]

            data         = activations[layer_names[pidx]]
            attn_maps, _ = _get_attention_maps(data, "content")
            N = min(attn_maps.shape[-1], len(res_names))

            cat_attn: dict[str, list[float]] = {}
            for h in high_heads:
                per_pos = attn_maps[h, :N, :N].mean(axis=0)
                for j, res in enumerate(res_names[:N]):
                    cat = AA_TO_CATEGORY.get(res, "Unknown")
                    cat_attn.setdefault(cat, []).append(float(per_pos[j]))

            means = [np.mean(cat_attn.get(c, [0.0])) for c in cats]
            bars  = ax.bar(range(len(cats)), means, color=colors,
                           edgecolor="white", linewidth=0.6)
            ax.spines["bottom"].set_color(pc)
            ax.spines["left"].set_color(pc)

            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha="right")
            ax.set_ylabel("Mean attention", fontsize=9)
            ax.set_title(
                f"Peak {panel_i+1} — Layer {layer_labels[pidx]}\n"
                f"({len(high_heads)} high-sem head{'s' if len(high_heads) > 1 else ''})",
                fontsize=10, color=pc,
            )
            ax.grid(axis="y", alpha=0.2)
            for bar, val in zip(bars, means):
                if val > 0.0005:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + max(means) * 0.015,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout(pad=1.5)
        if save_path_bars:
            fig_bars.savefig(save_path_bars, dpi=150, bbox_inches="tight")
        plt.show()

    return fig_line, fig_bars, layer_mean, peak_idxs


def plot_geometric_peaks(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    geo_scores: np.ndarray,
    layer_labels: list[int],
    save_path_line: Optional[str] = None,
    save_path_bars: Optional[str] = None,
    n_peaks: int = 3,
    geo_threshold: float = 0.4,
    figsize_line: Optional[tuple[float, float]] = None,
    figsize_bars: Optional[tuple[float, float]] = None,
    title_line: Optional[str] = None,
) -> tuple[plt.Figure, plt.Figure]:
    """Identify geometric-attention peaks and show which residue categories they target.

    Mirror of :func:`plot_semantic_peaks` but driven by ``geo_scores`` and the
    pairwise-bias attention component.

    Produces two independent figures:

    - **Line figure**: mean geometric KL score per layer.
    - **Bars figure**: for each peak, a bar chart of mean bias-attention received
      by each amino-acid category (computed from high-geo heads in that layer).

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    res_names : list[str]
    geo_scores : np.ndarray, shape ``[num_layers, num_heads]``
    layer_labels : list[int]
    save_path_line : str, optional
    save_path_bars : str, optional
    n_peaks : int
        Number of peaks to detect and show.  Default 3.
    geo_threshold : float
        Min normalised geo score for a head to be classified as "high-geometric".
    figsize_line : (width, height) in inches, optional
    figsize_bars : (width, height) in inches, optional
    title_line : str, optional
        Override the default title for the line-plot figure.

    Returns
    -------
    fig_line, fig_bars : matplotlib.figure.Figure
    layer_mean : np.ndarray, shape ``[num_layers]``
        Per-layer mean geometric score — the line that was plotted.
        Collect this across proteins to overlay on a single figure.
    peak_idxs : list[int]
        Layer indices identified as peaks (into ``layer_names``/``layer_labels``).
    """
    num_layers = len(layer_names)
    layer_mean = geo_scores.mean(axis=1)   # [L]

    # --- Detect peaks ---
    try:
        from scipy.signal import find_peaks as _fp
        peak_idxs, _ = _fp(layer_mean, distance=max(1, num_layers // 10))
        if len(peak_idxs) >= n_peaks:
            peak_idxs = peak_idxs[np.argsort(layer_mean[peak_idxs])[::-1][:n_peaks]]
        else:
            peak_idxs = np.argsort(layer_mean)[::-1][:n_peaks]
    except ImportError:
        peak_idxs = np.argsort(layer_mean)[::-1][:n_peaks]

    peak_idxs = sorted(int(p) for p in peak_idxs)
    peak_colors = ["#E07B7B", "#5BA85B", "#7B7BE0"][:n_peaks]

    with plt.rc_context(_ACADEMIC_RC):

        # ── Figure 1: line plot ──────────────────────────────────────────
        fs_line = figsize_line if figsize_line is not None else (8, 3.5)
        fig_line, ax_top = plt.subplots(figsize=fs_line)

        x = np.arange(num_layers)
        ax_top.plot(x, layer_mean, lw=1.8, color="#C05A2A", zorder=3,
                    label="Mean geo score")
        ax_top.fill_between(x, 0, layer_mean, alpha=0.12, color="#C05A2A")

        step = max(1, num_layers // 10)
        ax_top.set_xticks(range(0, num_layers, step))
        ax_top.set_xticklabels([str(layer_labels[p]) for p in range(0, num_layers, step)])
        ax_top.set_xlim(-0.5, num_layers - 0.5)
        ax_top.set_ylim(bottom=0)
        ax_top.set_xlabel("Layer")
        ax_top.set_ylabel("Mean geo KL score")
        _title = title_line if title_line is not None else (
            "Geometric attention importance across layers\n"
            "(peaks = layers with highest bias-driven routing)"
        )
        ax_top.set_title(_title)
        ax_top.legend(fontsize=9)
        ax_top.grid(True, alpha=0.2)
        plt.tight_layout(pad=1.5)
        if save_path_line:
            fig_line.savefig(save_path_line, dpi=150, bbox_inches="tight")
        plt.show()

        # ── Figure 2: residue-category bar charts ────────────────────────
        cats    = list(AA_CATEGORIES.keys()) + ["Unknown"]
        colors  = [_CATEGORY_COLORS[c] for c in cats]
        xlabels = [c.replace("+", "+\n").replace("-", "-\n") for c in cats]

        fs_bars = figsize_bars if figsize_bars is not None else (4.5 * n_peaks, 4)
        fig_bars, axes_bars = plt.subplots(
            1, n_peaks, figsize=fs_bars,
            gridspec_kw={"wspace": 0.4},
        )
        if n_peaks == 1:
            axes_bars = [axes_bars]

        for panel_i, (pidx, pc) in enumerate(zip(peak_idxs, peak_colors)):
            ax = axes_bars[panel_i]

            layer_geo  = geo_scores[pidx]
            high_heads = list(np.where(layer_geo >= geo_threshold)[0])
            if not high_heads:
                high_heads = [int(np.argmax(layer_geo))]

            data         = activations[layer_names[pidx]]
            attn_maps, _ = _get_attention_maps(data, "bias")
            N = min(attn_maps.shape[-1], len(res_names))

            cat_attn: dict[str, list[float]] = {}
            for h in high_heads:
                per_pos = attn_maps[h, :N, :N].mean(axis=0)
                for j, res in enumerate(res_names[:N]):
                    cat = AA_TO_CATEGORY.get(res, "Unknown")
                    cat_attn.setdefault(cat, []).append(float(per_pos[j]))

            means = [np.mean(cat_attn.get(c, [0.0])) for c in cats]
            bars  = ax.bar(range(len(cats)), means, color=colors,
                           edgecolor="white", linewidth=0.6)
            ax.spines["bottom"].set_color(pc)
            ax.spines["left"].set_color(pc)

            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha="right")
            ax.set_ylabel("Mean attention", fontsize=9)
            ax.set_title(
                f"Peak {panel_i+1} — Layer {layer_labels[pidx]}\n"
                f"({len(high_heads)} high-geo head{'s' if len(high_heads) > 1 else ''})",
                fontsize=10, color=pc,
            )
            ax.grid(axis="y", alpha=0.2)
            for bar, val in zip(bars, means):
                if val > 0.0005:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            val + max(means) * 0.015,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout(pad=1.5)
        if save_path_bars:
            fig_bars.savefig(save_path_bars, dpi=150, bbox_inches="tight")
        plt.show()

    return fig_line, fig_bars, layer_mean, peak_idxs


def plot_structure_vs_top_geo_bias(
    activations: dict,
    layer_names: list[str],
    ca_dist: np.ndarray,
    res_names: list[str],
    geo_scores: np.ndarray,
    save_path: Optional[str] = None,
    zoom: int = 80,
    figsize: Optional[tuple[float, float]] = None,
    title_prox: Optional[str] = None,
    title_bias: Optional[str] = None,
    suptitle: Optional[str] = None,
    save_path_prox: Optional[str] = None,
    save_path_bias: Optional[str] = None,
) -> plt.Figure:
    """Predicted-structure proximity vs the top geometric head's bias matrix.

    The diagonal is masked (set to NaN) to prevent it from dominating the colour
    scale.  The highest geometric head is taken from the **last** pairformer layer.

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    ca_dist : np.ndarray, shape ``[N, N]``
    res_names : list[str]
    geo_scores : np.ndarray, shape ``[num_layers, num_heads]``
    save_path : str, optional
    zoom : int
        Crop matrices to first *zoom* residues for readability.
    figsize : (width, height) in inches, optional
        Override the default ``(13, 5.5)``.
    title_prox : str, optional
        Override the default title for the proximity panel.
    title_bias : str, optional
        Override the default title for the bias attention panel.
    suptitle : str, optional
        Override the default figure-level suptitle.
    save_path_prox : str, optional
        If given, also save just the proximity panel to this path.
    save_path_bias : str, optional
        If given, also save just the bias attention panel to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    prox_mat : np.ndarray, shape ``[N, N]``
        Cα proximity matrix ``1/(d+1)`` that was plotted (diagonal NaN).
    bias_mat : np.ndarray, shape ``[N, N]``
        Top geo-head bias attention that was plotted (diagonal NaN).
    """
    last_idx  = len(layer_names) - 1
    best_head = int(np.argmax(geo_scores[last_idx]))
    best_geo  = float(geo_scores[last_idx, best_head])
    layer_depth = _layer_idx_from_name(layer_names[last_idx])

    attn_maps, _ = _get_attention_maps(activations[layer_names[last_idx]], "bias")
    N = min(attn_maps.shape[-1], len(res_names), zoom, ca_dist.shape[0])

    bias_mat = attn_maps[best_head, :N, :N].copy().astype(float)
    prox_mat = (1.0 / (ca_dist[:N, :N] + 1.0)).astype(float)

    # Mask diagonal so it doesn't anchor the colourscale
    np.fill_diagonal(bias_mat, np.nan)
    np.fill_diagonal(prox_mat, np.nan)

    _title_prox = title_prox if title_prox is not None else "Predicted structure\n(Cα proximity  1/(d+1))"
    _title_bias = title_bias if title_bias is not None else f"Pairwise-bias attention\n(layer {layer_depth}, head {best_head}, geo={best_geo:.2f})"
    _suptitle = suptitle if suptitle is not None else "Predicted 3-D structure vs geometric attention  (diagonal masked)"

    with plt.rc_context(_ACADEMIC_RC):
        fs = figsize if figsize is not None else (13, 5.5)
        fig, axes = plt.subplots(1, 2, figsize=fs,
                                 gridspec_kw={"wspace": 0.32})

        panels = [
            (prox_mat, _title_prox, "YlOrRd"),
            (bias_mat, _title_bias, "viridis"),
        ]
        for ax, (mat, title_str, cmap) in zip(axes, panels):
            im = ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
            ax.set_xlabel("Residue index")
            ax.set_ylabel("Residue index")
            ax.set_title(title_str, pad=8)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=9)

        fig.suptitle(_suptitle, fontsize=13, weight="bold", y=1.02)
        plt.tight_layout(pad=1.5)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

        if save_path_prox or save_path_bias:
            for panel_save, (mat, title_str, cmap) in zip(
                [save_path_prox, save_path_bias],
                [(prox_mat, _title_prox, "YlOrRd"), (bias_mat, _title_bias, "viridis")],
            ):
                if not panel_save:
                    continue
                fig_p, ax_p = plt.subplots(figsize=(6.5, 5.5))
                im_p = ax_p.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest")
                ax_p.set_xlabel("Residue index")
                ax_p.set_ylabel("Residue index")
                ax_p.set_title(title_str, pad=8)
                cbar_p = fig_p.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)
                cbar_p.ax.tick_params(labelsize=9)
                plt.tight_layout(pad=1.5)
                fig_p.savefig(panel_save, dpi=150, bbox_inches="tight")
                plt.close(fig_p)

    return fig, prox_mat, bias_mat


def plot_bias_sampled_layers(
    activations: dict,
    layer_names: list[str],
    geo_scores: np.ndarray,
    res_names: list[str],
    save_path: Optional[str] = None,
    zoom: int = 80,
    n_samples: int = 4,
    figsize: Optional[tuple[float, float]] = None,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Bias attention matrices for the top geometric head in *n_samples* evenly-spaced layers.

    Parameters
    ----------
    activations : dict
    layer_names : list[str]
    geo_scores : np.ndarray, shape ``[num_layers, num_heads]``
    res_names : list[str]
    save_path : str, optional
    zoom : int
        Crop to the first *zoom* residues.
    n_samples : int
        Number of layers to sample.  Default 4.
    figsize : (width, height) in inches, optional
        Override the default ``(4.5 * n_samples, 5)``.
    suptitle : str, optional
        Override the default figure-level suptitle.

    Returns
    -------
    fig : matplotlib.figure.Figure
    sampled_mats : dict[int, np.ndarray]
        ``{layer_depth: bias_mat}`` for each sampled layer, where ``bias_mat``
        has shape ``[N, N]``.  Collect across proteins to compare panels.
    """
    num_layers = len(layer_names)
    if n_samples == 1:
        sample_idxs = [num_layers - 1]
    else:
        sample_idxs = [
            int(round(i * (num_layers - 1) / (n_samples - 1)))
            for i in range(n_samples)
        ]

    sampled_mats: dict[int, np.ndarray] = {}

    with plt.rc_context(_ACADEMIC_RC):
        fs = figsize if figsize is not None else (4.5 * n_samples, 5)
        fig, axes = plt.subplots(1, n_samples,
                                 figsize=fs,
                                 gridspec_kw={"wspace": 0.3})
        if n_samples == 1:
            axes = [axes]

        for ax, lidx in zip(axes, sample_idxs):
            best_head   = int(np.argmax(geo_scores[lidx]))
            geo_val     = float(geo_scores[lidx, best_head])
            layer_depth = _layer_idx_from_name(layer_names[lidx])

            attn_maps, _ = _get_attention_maps(activations[layer_names[lidx]], "bias")
            N   = min(attn_maps.shape[-1], len(res_names), zoom)
            mat = attn_maps[best_head, :N, :N]
            sampled_mats[layer_depth] = mat

            im = ax.imshow(mat, cmap="viridis", aspect="auto",
                           interpolation="nearest", vmin=0)
            ax.set_title(
                f"Layer {layer_depth}\nHead {best_head}  (geo={geo_val:.2f})",
                pad=8, fontsize=11,
            )
            ax.set_xlabel("Residue")
            ax.set_ylabel("Residue")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=8)

        _suptitle = suptitle if suptitle is not None else "Geometric attention — top geo head per sampled layer"
        fig.suptitle(_suptitle, fontsize=13, weight="bold")
        plt.tight_layout(pad=1.5)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig, sampled_mats


# ---------------------------------------------------------------------------
# 13. Geo-head bias–structure correlation heatmap
# ---------------------------------------------------------------------------

def plot_geo_head_correlation_heatmap(
    activations: dict,
    layer_names: list[str],
    ca_dist: np.ndarray,
    res_names: list[str],
    geo_scores: np.ndarray,
    geo_threshold: float = 0.5,
    save_path: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Spearman correlation between each geo head's bias attention and Cα proximity.

    Only heads whose ``geo_scores`` exceeds ``geo_threshold`` are evaluated;
    the rest are blacked out on the heatmap.

    Parameters
    ----------
    activations : dict
        Single-step activations (output of :func:`get_recycling_step`).
    layer_names : list[str]
        Ordered main-pairformer layer names.
    ca_dist : np.ndarray, shape ``[N, N]``
        Pairwise Cα distances in Å.
    res_names : list[str]
    geo_scores : np.ndarray, shape ``[num_layers, num_heads]``
        Geometric scores (ratio or raw-normalised) for thresholding.
    geo_threshold : float
        Heads with ``geo_scores <= geo_threshold`` are blacked out.  Default 0.5.
    save_path : str, optional
    figsize : (width, height) in inches, optional
    title : str, optional
        Override the default axis title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    corr_mat : np.ndarray, shape ``[num_layers, num_heads]``
        Spearman r values; NaN for heads below threshold.
    """
    from scipy.stats import spearmanr

    num_layers = len(layer_names)
    num_heads  = geo_scores.shape[1]
    corr_mat   = np.full((num_layers, num_heads), np.nan, dtype=np.float32)

    N_dist = ca_dist.shape[0]
    prox   = 1.0 / (ca_dist + 1.0)

    for i, name in enumerate(layer_names):
        data         = activations[name]
        attn_maps, _ = _get_attention_maps(data, "bias")
        N = min(attn_maps.shape[-1], len(res_names), N_dist)

        prox_flat = prox[:N, :N].copy()
        np.fill_diagonal(prox_flat, np.nan)
        mask     = ~np.isnan(prox_flat)
        prox_vec = prox_flat[mask]

        for h in range(num_heads):
            if geo_scores[i, h] <= geo_threshold:
                continue  # leave NaN → blacked out

            bias_mat = attn_maps[h, :N, :N].copy().astype(float)
            np.fill_diagonal(bias_mat, np.nan)
            bias_vec = bias_mat[mask]

            r, _ = spearmanr(prox_vec, bias_vec)
            corr_mat[i, h] = float(r)

    layer_labels = [_layer_idx_from_name(n) for n in layer_names]

    with plt.rc_context(_ACADEMIC_RC):
        fs = figsize if figsize is not None else (
            max(7, num_heads * 0.5 + 2),
            max(5, num_layers * 0.18 + 2),
        )
        fig, ax = plt.subplots(figsize=fs)

        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("black")

        masked = np.ma.masked_invalid(corr_mat)
        im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto",
                       interpolation="nearest")

        step = max(1, num_layers // 10)
        ax.set_yticks(range(0, num_layers, step))
        ax.set_yticklabels([str(layer_labels[p]) for p in range(0, num_layers, step)])
        ax.set_xticks(range(num_heads))
        ax.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=8)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        _title = title if title is not None else (
            f"Bias–structure Spearman r  (geo heads, threshold={geo_threshold:.2f})\n"
            "Black = below threshold"
        )
        ax.set_title(_title, pad=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Spearman r", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout(pad=1.5)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig, corr_mat


# ---------------------------------------------------------------------------
# 14. Bias-correlation gallery
# ---------------------------------------------------------------------------

def plot_bias_correlation_gallery(
    activations: dict,
    layer_names: list[str],
    res_names: list[str],
    corr_mat: np.ndarray,
    geo_threshold: float = 0.5,
    zoom: int = 80,
    save_path: Optional[str] = None,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> tuple[plt.Figure, dict]:
    """Correlation heatmap (top) with three representative bias matrices (bottom).

    The three heads are selected from among geo heads above *geo_threshold*:

    - **High**: head with the highest Spearman r
    - **Mid**: head closest to the median r value
    - **Low**: head with the lowest Spearman r

    Dashed connector lines link each selected cell in the heatmap to its
    bias-attention matrix below.

    Parameters
    ----------
    activations : dict
        Single-step activations from :func:`get_recycling_step`.
    layer_names : list[str]
    res_names : list[str]
    corr_mat : np.ndarray, shape ``[num_layers, num_heads]``
        Output of :func:`plot_geo_head_correlation_heatmap`; NaN = below threshold.
    geo_threshold : float
        Only used for the figure title annotation.
    zoom : int
        Crop bias matrices to first *zoom* residues.
    save_path : str, optional
    figsize : (width, height) in inches, optional
        Default ``(15, 11)``.
    title : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    selected : dict
        ``{'high': (layer_i, head_i, r), 'mid': ..., 'low': ...}``
    """
    # ── Select three representative heads ─────────────────────────────
    valid_mask = ~np.isnan(corr_mat)
    if valid_mask.sum() < 3:
        raise ValueError(
            f"Fewer than 3 geo heads above threshold ({valid_mask.sum()} found); "
            "cannot build gallery."
        )

    valid_idxs  = np.argwhere(valid_mask)           # [K, 2] — (layer_i, head_i)
    valid_vals  = corr_mat[valid_mask]              # [K]
    sorted_ord  = np.argsort(valid_vals)[::-1]     # descending by r
    sorted_idxs = valid_idxs[sorted_ord]
    sorted_vals = valid_vals[sorted_ord]

    n_valid   = len(sorted_vals)
    mid_rank  = int(np.argmin(np.abs(sorted_vals - np.median(sorted_vals))))
    sel_ranks  = [0, mid_rank, n_valid - 1]

    selected: dict = {}
    for rank, tag in zip(sel_ranks, ["high", "mid", "low"]):
        li, hi = sorted_idxs[rank]
        selected[tag] = (int(li), int(hi), float(sorted_vals[rank]))

    sel_labels = ["Close Contacts", "Sequential Encoding", "Long-Range Contacts"]
    sel_colors = ["#2E7D32", "#E65100", "#C62828"]   # green / orange / red

    num_layers, num_heads = corr_mat.shape
    layer_labels = [_layer_idx_from_name(n) for n in layer_names]

    with plt.rc_context(_ACADEMIC_RC):
        fs  = figsize if figsize is not None else (12, 11)
        fig = plt.figure(figsize=fs)

        # Two rows: heatmap (same width as one matrix, centred) + 3 matrices.
        # Row 0 col 1 only is used; cols 0 and 2 are empty to leave space for
        # the dashed connector lines to the outer matrices.
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[1.0, 1.0],
            hspace=0.60, wspace=0.38,
            left=0.07, right=0.96, top=0.90, bottom=0.06,
        )

        # ── Top row: correlation heatmap (centre column only) ─────────
        ax_heat = fig.add_subplot(gs[0, 1])

        cmap_heat = plt.cm.RdBu_r.copy()
        cmap_heat.set_bad("black")
        masked = np.ma.masked_invalid(corr_mat)
        im = ax_heat.imshow(masked, cmap=cmap_heat, vmin=-1, vmax=1,
                            aspect="auto", interpolation="nearest")

        step = max(1, num_layers // 10)
        ax_heat.set_yticks(range(0, num_layers, step))
        ax_heat.set_yticklabels(
            [str(layer_labels[p]) for p in range(0, num_layers, step)]
        )
        ax_heat.set_xticks(range(num_heads))
        ax_heat.set_xticklabels([f"H{h}" for h in range(num_heads)], fontsize=9)
        ax_heat.set_xlabel("Attention head", labelpad=6)
        ax_heat.set_ylabel("Layer", labelpad=6)

        _title = title if title is not None else (
            f"Bias–structure Spearman r  (geo threshold={geo_threshold:.0%};  "
            "black = below threshold)"
        )
        ax_heat.set_title(_title, pad=10)

        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.01)
        cbar.set_label("Spearman r", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # Highlight selected cells with coloured rectangles
        for (li, hi, _), color in zip(selected.values(), sel_colors):
            rect = plt.Rectangle(
                (hi - 0.5, li - 0.5), 1, 1,
                linewidth=2.5, edgecolor=color, facecolor="none", zorder=5,
            )
            ax_heat.add_patch(rect)

        # ── Bottom row: one bias matrix per selected head ─────────────
        bias_axes = [fig.add_subplot(gs[1, col]) for col in range(3)]

        for ax_b, (tag, (li, hi, r_val)), lbl, color in zip(
            bias_axes, selected.items(), sel_labels, sel_colors
        ):
            data         = activations[layer_names[li]]
            attn_maps, _ = _get_attention_maps(data, "bias")
            N   = min(attn_maps.shape[-1], len(res_names), zoom)
            mat = attn_maps[hi, :N, :N]

            im_b = ax_b.imshow(mat, cmap="viridis", aspect="auto",
                               interpolation="nearest", vmin=0)
            ax_b.set_title(
                f"{lbl}\nLayer {layer_labels[li]}, Head {hi}   r = {r_val:.3f}",
                fontsize=10, color=color, pad=8,
            )
            ax_b.set_xlabel("Residue", fontsize=9)
            ax_b.set_ylabel("Residue", fontsize=9)

            for spine in ax_b.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)

            cb = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=8)

            # ── Dashed connector line: heatmap cell → top-centre of matrix
            # coordsA="data": x = head col, y = layer row in the heatmap image
            con = mpatches.ConnectionPatch(
                xyA=(hi, li),
                xyB=(0.5, 1.0),
                coordsA="data",
                coordsB="axes fraction",
                axesA=ax_heat,
                axesB=ax_b,
                color=color,
                lw=1.5,
                ls=(0, (5, 4)),      # dashed
                alpha=0.80,
                zorder=10,
                clip_on=False,
                shrinkB=42,          # stop before the title text
            )
            fig.add_artist(con)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    return fig, selected
