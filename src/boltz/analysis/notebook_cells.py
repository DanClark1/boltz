"""
Colab notebook cells for Boltz interpretability analysis.

Copy each section into its own Colab cell in order.
All cells assume the standard boltz Colab setup:

    !git clone "https://github.com/DanClark1/boltz.git"
    !cp -r "boltz" "boltz_repo"
    !pip install -e boltz_repo
    !rm -rf "boltz"
    !cp -r "boltz_repo/src/boltz" "./boltz"
    !rm -rf "boltz_repo"

──────────────────────────────────────────────────────────────────────────────
CELL 1 – Setup & hook registration (replaces your original prediction cell)
──────────────────────────────────────────────────────────────────────────────

import torch
import boltz
from boltz.analysis.interp import (
    register_metadata_hook,
    decode_res_types,
)

file_name = "boltz_activations_prot_no_msa"
yaml      = "/content/drive/MyDrive/gdl/examples/prot_no_msa.yaml"

from boltz.model.layers.pairformer import AttentionPairBias

# 1. Load model
model = boltz.load_model("boltz1", device="cuda", use_kernels=False)

# 2. Metadata hook – captures residue type info from the batch dict
metadata, meta_handle = register_metadata_hook(model)

# 3. Activation storage
activations = {}

def get_hook(layer_name, component_name):
    def hook(module, input, output):
        if layer_name not in activations:
            activations[layer_name] = {}
        activations[layer_name][component_name] = output.detach().cpu().clone()
    return hook

# 4. Register activation hooks
print("Registering hooks...")
hook_handles = []
for name, module in model.named_modules():
    if "AttentionPairBias" in str(type(module)):
        print(f"  Hooking: {name}")
        h1 = module.proj_q.register_forward_hook(get_hook(name, 'q'))
        h2 = module.proj_k.register_forward_hook(get_hook(name, 'k'))
        h3 = module.proj_z.register_forward_hook(get_hook(name, 'bias'))
        h4 = module.proj_o.register_forward_hook(get_hook(name, 'o'))
        hook_handles.extend([h1, h2, h3, h4])

# 5. Run prediction
print("Running inference...")
results = boltz.predict(
    model,
    yaml,
    use_msa_server=True,
    diffusion_samples=1,
)

# 6. Decode residue names from the captured batch metadata
res_names = decode_res_types(
    metadata['res_type'],
    metadata.get('token_pad_mask'),
)
print(f"Sequence length: {len(res_names)} tokens")
print("First 20 residues:", res_names[:20])

# 7. Save everything
save_path = "/content/drive/MyDrive/" + file_name + ".pt"
torch.save(activations, save_path)
torch.save(results[0]['coords'], '/content/drive/MyDrive/' + file_name + '_coords.pt')
torch.save({'res_names': res_names, 'metadata': metadata},
           '/content/drive/MyDrive/' + file_name + '_meta.pt')
print(f"Saved activations → {save_path}")

# 8. Cleanup
for h in hook_handles:
    h.remove()
meta_handle.remove()
print("Done.")


──────────────────────────────────────────────────────────────────────────────
CELL 2 – Load saved data & build Cα distance matrix
──────────────────────────────────────────────────────────────────────────────

import torch
import numpy as np
import re
from boltz.analysis.interp import (
    compute_ca_coords,
    compute_distance_matrix,
)

file_name = "boltz_activations_prot_no_msa"

activations = torch.load("/content/drive/MyDrive/" + file_name + ".pt",
                          map_location="cpu")
coords      = torch.load("/content/drive/MyDrive/" + file_name + "_coords.pt",
                          map_location="cpu")
meta        = torch.load("/content/drive/MyDrive/" + file_name + "_meta.pt",
                          map_location="cpu")
res_names   = meta['res_names']

def get_layer_idx(name):
    m = re.search(r"layers\\.(\d+)", name)
    return int(m.group(1)) if m else -1

layer_names = sorted(
    [k for k in activations if "pairformer_module" in k],
    key=get_layer_idx,
)
layer_labels = [get_layer_idx(n) for n in layer_names]
num_layers   = len(layer_names)
print(f"Loaded {num_layers} layers, sequence length {len(res_names)}")

# Build Cα coordinate array and pairwise distance matrix
ca_coords  = compute_ca_coords(coords, res_names, sample_idx=0)
ca_dist    = compute_distance_matrix(ca_coords)
print(f"Cα distance matrix: {ca_dist.shape}")


──────────────────────────────────────────────────────────────────────────────
CELL 3 – (Optional) Alternative: load residue names from processed structure
          Use this if you did not capture the metadata hook, or as a check.
──────────────────────────────────────────────────────────────────────────────

# from boltz.analysis.interp import load_structure_residues, compute_ca_coords
#
# # 'prot_no_msa' is the stem of the yaml file; adjust as needed.
# res_names_from_struct, atom_centers = load_structure_residues(
#     'prot_no_msa/processed', 'prot_no_msa'
# )
# print(res_names_from_struct[:20])
#
# ca_coords = compute_ca_coords(coords, res_names_from_struct)
# ca_dist   = compute_distance_matrix(ca_coords)


──────────────────────────────────────────────────────────────────────────────
CELL 4 – Residue type attention analysis
         (which amino-acid types get attended to by bias vs content?)
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import plot_residue_type_attention

plot_residue_type_attention(
    activations,
    layer_names,
    res_names,
    fig_title=f"Residue Type Attention — {file_name}",
    save_path=f"/content/drive/MyDrive/{file_name}_residue_type_attention.png",
)


──────────────────────────────────────────────────────────────────────────────
CELL 5 – Top attended positions for a specific head
         (shows exactly which sequence positions are attended to, coloured
          by biochemical category)
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import plot_top_attended_residues_for_head

# Adjust layer_idx and head_idx to the head you want to inspect.
# E.g. a "High Geo – Low Sem" head from your KL scatter plot.
plot_top_attended_residues_for_head(
    activations,
    layer_names,
    res_names,
    layer_idx=0,     # index into layer_names list
    head_idx=0,
    top_k=15,
    save_path=f"/content/drive/MyDrive/{file_name}_top_residues_L0H0.png",
)


──────────────────────────────────────────────────────────────────────────────
CELL 6 – Bias vs predicted structure for a specific head
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import plot_bias_vs_structure

plot_bias_vs_structure(
    activations,
    layer_names,
    ca_dist,
    res_names,
    layer_idx=0,
    head_idx=0,
    zoom=80,
    save_path=f"/content/drive/MyDrive/{file_name}_structure_comparison_L0H0.png",
)


──────────────────────────────────────────────────────────────────────────────
CELL 7 – Per-layer, per-head structure correlation heatmaps
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import (
    compute_layer_structure_correlations,
    plot_structure_correlation_heatmap,
)

bias_corr    = compute_layer_structure_correlations(
    activations, layer_names, ca_dist, component='bias')
content_corr = compute_layer_structure_correlations(
    activations, layer_names, ca_dist, component='content')

print(f"Bias–structure mean corr: {bias_corr.mean():.3f}")
print(f"Content–structure mean corr: {content_corr.mean():.3f}")

plot_structure_correlation_heatmap(
    bias_corr,
    content_corr,
    layer_labels,
    save_path=f"/content/drive/MyDrive/{file_name}_structure_corr_heatmap.png",
)


──────────────────────────────────────────────────────────────────────────────
CELL 8 – Full extended analysis (runs Cells 7 + KL heatmaps together)
         Run this AFTER your original KL permutation analysis cell so that
         geo_scores and sem_scores are already defined.
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import plot_full_extended_analysis

fig, bias_corr, content_corr = plot_full_extended_analysis(
    activations,
    layer_names,
    ca_dist,
    res_names,
    geo_scores,   # from original KL analysis (normalised, shape [L, H])
    sem_scores,   # from original KL analysis (normalised, shape [L, H])
    layer_labels,
    save_path=f"/content/drive/MyDrive/{file_name}_extended_analysis.png",
)


──────────────────────────────────────────────────────────────────────────────
CELL 9 – Loop: bias-vs-structure for all 4 head categories
         (Plug into the existing category analysis after your KL analysis)
──────────────────────────────────────────────────────────────────────────────

from boltz.analysis.interp import plot_bias_vs_structure, plot_top_attended_residues_for_head

for cat_name, indices in valid_cats.items():   # valid_cats from your KL cell
    safe = cat_name.split('\\n')[1].strip('()').replace(' ', '_')
    for row, (layer_idx, head_idx) in enumerate(indices[:3]):  # first 3 examples
        plot_bias_vs_structure(
            activations, layer_names, ca_dist, res_names,
            layer_idx=layer_idx, head_idx=head_idx,
            zoom=80,
            save_path=f"/content/drive/MyDrive/{file_name}_{safe}_L{layer_idx}H{head_idx}_struct.png",
        )
        plot_top_attended_residues_for_head(
            activations, layer_names, res_names,
            layer_idx=layer_idx, head_idx=head_idx,
            top_k=15,
            save_path=f"/content/drive/MyDrive/{file_name}_{safe}_L{layer_idx}H{head_idx}_residues.png",
        )


──────────────────────────────────────────────────────────────────────────────
CELL 10 – Inference pipeline  (run once per example to capture activations)
          Loops over four YAML inputs and saves recycling-aware activations,
          coordinates and residue metadata to Drive.
──────────────────────────────────────────────────────────────────────────────

import os, re, torch
import boltz
from boltz.analysis.interp import register_metadata_hook, decode_res_types, get_recycling_step

DRIVE    = "/content/drive/MyDrive"
YAML_DIR = f"{DRIVE}/gdl/examples"
OUT_DIR  = f"{DRIVE}/final_experiments"
os.makedirs(OUT_DIR, exist_ok=True)

EXAMPLES = ["prot_no_msa", "prot", "multimer", "ligand"]

# Load model once
model = boltz.load_model("boltz1", device="cuda", use_kernels=False)

import shutil, tempfile

for example in EXAMPLES:
    save_base = f"{OUT_DIR}/{example}/raw"
    os.makedirs(save_base, exist_ok=True)

    acts_path = f"{save_base}/activations.pt"
    if os.path.exists(acts_path):
        print(f"[{example}] Already captured – skipping inference.")
        continue

    print(f"\n{'='*60}\n[{example}] Running inference …")

    metadata, meta_handle = register_metadata_hook(model)
    activations = {}

    def get_hook(layer_name, component_name):
        def hook(module, input, output):
            # Overwrite on every recycling step (keep only the latest).
            # Stored as a 1-element list so get_recycling_step(step=-1) still works.
            # fp16 halves memory; precision is fine for KL/attention analysis.
            if layer_name not in activations:
                activations[layer_name] = {}
            activations[layer_name][component_name] = [
                output.detach().cpu().half()
            ]
        return hook

    hook_handles = []
    for name, module in model.named_modules():
        if "AttentionPairBias" in str(type(module)):
            hook_handles += [
                module.proj_q.register_forward_hook(get_hook(name, "q")),
                module.proj_k.register_forward_hook(get_hook(name, "k")),
                module.proj_z.register_forward_hook(get_hook(name, "bias")),
                # proj_o is not used in any analysis — skip to save memory.
            ]

    yaml_path = f"{YAML_DIR}/{example}.yaml"

    # Route boltz output to /tmp so processed inputs, MSA files and
    # prediction CIFs don't accumulate on the local Colab disk.
    tmp_out = f"/tmp/boltz_{example}"
    try:
        results = boltz.predict(model, yaml_path, out_dir=tmp_out,
                                use_msa_server=True,
                                recycling_steps=3, diffusion_samples=1)

        res_names = decode_res_types(metadata["res_type"],
                                      metadata.get("token_pad_mask"))
        print(f"  Sequence length: {len(res_names)} tokens")

        torch.save(activations,          f"{save_base}/activations.pt")
        torch.save(results[0]["coords"], f"{save_base}/coords.pt")
        torch.save({"res_names": res_names}, f"{save_base}/meta.pt")
        print(f"  Saved to {save_base}")
    finally:
        # Always clean up the local boltz scratch dir
        shutil.rmtree(tmp_out, ignore_errors=True)
        for h in hook_handles: h.remove()
        meta_handle.remove()
        activations.clear()

print("\nInference complete.")


──────────────────────────────────────────────────────────────────────────────
CELL 11 – Analysis & figure pipeline
          Loads saved activations for each example and generates all plots.
          Per-example figures → final_experiments/<example>/
          Multi-protein overlays → final_experiments/combined/
──────────────────────────────────────────────────────────────────────────────

import os, re, numpy as np, torch, matplotlib.pyplot as plt
import gc
from boltz.analysis.interp import (
    get_recycling_step,
    compute_ca_coords,
    compute_distance_matrix,
    run_kl_analysis,
    plot_kl_heatmaps_separate,
    plot_kl_heatmap_combined,
    plot_semantic_peaks,
    plot_geometric_peaks,
    plot_geo_head_correlation_heatmap,
    plot_bias_correlation_gallery,
    plot_structure_vs_top_geo_bias,
    plot_bias_sampled_layers,
)

DRIVE      = "/content/drive/MyDrive"
OUT_ROOT   = f"{DRIVE}/final_experiments"
GLOBAL_DIR = f"{OUT_ROOT}/combined"
EXAMPLES   = ["prot_no_msa", "prot", "multimer", "cyclic_prot", "ligand"]
# EXAMPLES = ["ligand"]
os.makedirs(GLOBAL_DIR, exist_ok=True)

EXAMPLE_COLORS = {
    "prot_no_msa": "#5B8DB8",
    "prot":        "#E07B7B",
    "multimer":    "#5BA85B",
    "cyclic_prot": "#C05A2A",
}



EXAMPLE_LABELS = {
    "prot_no_msa": "α3D (Single Seq)",
    "prot":        "α3D (with MSA)",
    "multimer":    "Heterodimer ",
    "cyclic_prot": "Cyclic Peptide",
    "ligand":      "Protein-Ligand"
}

FIG_SIZES = {
    "rq1_heatmap":        None,
    "rq1_combined":       None,
    "rq1_sem_peaks_line": (5, 5),
    "rq1_sem_peaks_bars": None,
    "rq1_geo_peaks_line": (5, 5),
    "rq1_geo_peaks_bars": None,
    "rq1_corr":           None,
    "rq1_corr_gallery":   None,
    "rq2_struct":         None,
    "rq2_sampled":        None,
    "overlay":            (14, 8),
}

GEO_THRESHOLD = 0.5

def _layer_idx(name):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else -1


def process_example(example):
    """Run full analysis for one protein; return only the lightweight overlay lines."""
    raw_dir = f"{OUT_ROOT}/{example}/raw"
    out_dir = f"{OUT_ROOT}/{example}"
    os.makedirs(out_dir, exist_ok=True)

    acts_path = f"{raw_dir}/activations.pt"
    if not os.path.exists(acts_path):
        print(f"[{example}] No activations found – run Cell 10 first.")
        return None

    label = EXAMPLE_LABELS.get(example, example)

    # Helper: build a save path with the structure name embedded in the filename
    def sp(stem):
        return f"{out_dir}/{example}_{stem}.png"

    print(f"\n{'='*60}\n[{example}] Loading data …")
    all_acts  = torch.load(acts_path,               map_location="cpu")
    coords    = torch.load(f"{raw_dir}/coords.pt",  map_location="cpu")
    meta      = torch.load(f"{raw_dir}/meta.pt",    map_location="cpu")
    res_names = meta["res_names"]

    layer_names  = sorted(
        [k for k in all_acts if k.startswith("pairformer_module.")],
        key=_layer_idx,
    )
    layer_labels = [_layer_idx(n) for n in layer_names]
    print(f"  Layers: {len(layer_names)}   Residues: {len(res_names)}")

    acts      = get_recycling_step(all_acts, step=-1)
    del all_acts

    ca_coords = compute_ca_coords(coords, res_names, sample_idx=0)
    ca_dist   = compute_distance_matrix(ca_coords)
    del coords, ca_coords

    # ── KL analysis ───────────────────────────────────────────────────
    print("  Running KL analysis …", flush=True)
    geo_raw, sem_raw = run_kl_analysis(acts, layer_names, num_trials=5)

    total     = geo_raw + sem_raw + 1e-10
    geo_ratio = geo_raw / total
    sem_ratio = sem_raw / total

    geo_p95  = np.percentile(geo_raw, 95)
    sem_p95  = np.percentile(sem_raw, 95)
    geo_norm = np.clip(geo_raw / (geo_p95 + 1e-10), 0, 1)
    sem_norm = np.clip(sem_raw / (sem_p95 + 1e-10), 0, 1)

    print(f"  ratio  geo={geo_ratio.mean():.3f}  sem={sem_ratio.mean():.3f}")
    print(f"  raw    geo={geo_norm.mean():.3f}  sem={sem_norm.mean():.3f}")

    # ── RQ1-1  Heatmaps – ratio & raw ────────────────────────────────
    for tag, geo_s, sem_s in [("ratio", geo_ratio, sem_ratio),
                               ("raw",   geo_norm,  sem_norm)]:
        plot_kl_heatmaps_separate(
            geo_s, sem_s, layer_labels,
            save_geo=sp(f"rq1_{tag}_geo_heatmap"),
            save_sem=sp(f"rq1_{tag}_sem_heatmap"),
            figsize=FIG_SIZES["rq1_heatmap"],
            title_geo=f"Geometric importance ({tag})  —  {label}",
            title_sem=f"Semantic importance ({tag})  —  {label}",
        )
        plot_kl_heatmap_combined(
            geo_s, sem_s, layer_labels,
            save_path=sp(f"rq1_{tag}_combined_heatmap"),
            figsize=FIG_SIZES["rq1_combined"],
            suptitle=f"Geometric vs Semantic importance ({tag})  —  {label}",
        )

    # ── RQ1-3  Semantic peaks – ratio & raw ──────────────────────────
    _, _, sem_line_ratio, _ = plot_semantic_peaks(
        acts, layer_names, res_names, sem_ratio, layer_labels,
        n_peaks=3, sem_threshold=0.4,
        save_path_line=sp("rq1_ratio_sem_peaks_line"),
        save_path_bars=sp("rq1_ratio_sem_peaks_bars"),
        figsize_line=FIG_SIZES["rq1_sem_peaks_line"],
        figsize_bars=FIG_SIZES["rq1_sem_peaks_bars"],
        title_line=f"Semantic routing across layers (ratio)  —  {label}",
    )
    _, _, sem_line_raw, _ = plot_semantic_peaks(
        acts, layer_names, res_names, sem_norm, layer_labels,
        n_peaks=3, sem_threshold=0.4,
        save_path_line=sp("rq1_raw_sem_peaks_line"),
        save_path_bars=sp("rq1_raw_sem_peaks_bars"),
        figsize_line=FIG_SIZES["rq1_sem_peaks_line"],
        figsize_bars=FIG_SIZES["rq1_sem_peaks_bars"],
        title_line=f"Semantic routing across layers (raw norm.)  —  {label}",
    )

    # ── RQ1-4  Geometric peaks – ratio & raw ─────────────────────────
    _, _, geo_line_ratio, _ = plot_geometric_peaks(
        acts, layer_names, res_names, geo_ratio, layer_labels,
        n_peaks=3, geo_threshold=0.4,
        save_path_line=sp("rq1_ratio_geo_peaks_line"),
        save_path_bars=sp("rq1_ratio_geo_peaks_bars"),
        figsize_line=FIG_SIZES["rq1_geo_peaks_line"],
        figsize_bars=FIG_SIZES["rq1_geo_peaks_bars"],
        title_line=f"Geometric routing across layers (ratio)  —  {label}",
    )
    _, _, geo_line_raw, _ = plot_geometric_peaks(
        acts, layer_names, res_names, geo_norm, layer_labels,
        n_peaks=3, geo_threshold=0.4,
        save_path_line=sp("rq1_raw_geo_peaks_line"),
        save_path_bars=sp("rq1_raw_geo_peaks_bars"),
        figsize_line=FIG_SIZES["rq1_geo_peaks_line"],
        figsize_bars=FIG_SIZES["rq1_geo_peaks_bars"],
        title_line=f"Geometric routing across layers (raw norm.)  —  {label}",
    )

    # ── RQ1-5  Bias–structure correlation heatmap ─────────────────────
    _, corr_mat = plot_geo_head_correlation_heatmap(
        acts, layer_names, ca_dist, res_names, geo_ratio,
        geo_threshold=GEO_THRESHOLD,
        save_path=sp("rq1_geo_corr_heatmap"),
        figsize=FIG_SIZES["rq1_corr"],
        title=f"Bias–structure correlation (geo heads >{GEO_THRESHOLD:.0%})  —  {label}",
    )

    # ── RQ1-6  Bias correlation gallery ───────────────────────────────
    if corr_mat is not None and (~np.isnan(corr_mat)).sum() >= 3:
        prox_mat = 1.0 / (ca_dist + 1.0)
        plot_bias_correlation_gallery(
            acts, layer_names, res_names, corr_mat,
            prox_mat=prox_mat,
            geo_threshold=GEO_THRESHOLD,
            zoom=min(80, len(res_names)),
            save_path=sp("rq1_corr_gallery"),
            figsize=FIG_SIZES["rq1_corr_gallery"],
            title=f"Bias–structure Spearman r  —  {label}",
            contact_title=f"Ground-truth Cα proximity  —  {label}",
        )

    # ── RQ2-1  Structure proximity vs bias attention ───────────────────
    plot_structure_vs_top_geo_bias(
        acts, layer_names, ca_dist, res_names, geo_ratio,
        zoom=min(80, len(res_names)),
        save_path=sp("rq2_structure_vs_bias_combined"),
        save_path_prox=sp("rq2_structure_proximity"),
        save_path_bias=sp("rq2_geo_bias_attention"),
        figsize=FIG_SIZES["rq2_struct"],
        suptitle=f"Structure proximity vs geometric attention  —  {label}",
    )

    # ── RQ2-2  Sampled bias matrices ──────────────────────────────────
    plot_bias_sampled_layers(
        acts, layer_names, geo_ratio, res_names,
        zoom=min(80, len(res_names)),
        n_samples=4,
        save_path=sp("rq2_bias_sampled"),
        figsize=FIG_SIZES["rq2_sampled"],
        suptitle=f"Geometric attention across model depth  —  {label}",
    )

    # ── Summary text file ─────────────────────────────────────────────
    num_heads   = geo_raw.shape[1]
    valid_corrs = corr_mat[~np.isnan(corr_mat)]
    n_geo_heads = int((geo_ratio > GEO_THRESHOLD).sum())

    # Best correlated head (highest Spearman r)
    if valid_corrs.size > 0:
        best_flat   = int(np.nanargmax(corr_mat))
        best_layer  = layer_labels[best_flat // num_heads]
        best_head   = best_flat % num_heads
        best_r      = float(np.nanmax(corr_mat))
        corr_mean   = float(valid_corrs.mean())
        corr_std    = float(valid_corrs.std())
    else:
        best_layer = best_head = best_r = corr_mean = corr_std = float("nan")

    summary_lines = [
        f"Structure summary: {label} ({example})",
        f"{'='*50}",
        f"Residues / tokens : {len(res_names)}",
        f"Pairformer layers  : {len(layer_names)}",
        f"Attention heads    : {num_heads}",
        f"",
        f"── KL divergence (raw) ──────────────────────────",
        f"  Mean geo KL        : {geo_raw.mean():.4f}",
        f"  Mean sem KL        : {sem_raw.mean():.4f}",
        f"  Geo / (geo+sem)    : {geo_ratio.mean():.4f}",
        f"  Sem / (geo+sem)    : {sem_ratio.mean():.4f}",
        f"",
        f"── Raw KL (95th-pct normalised) ─────────────────",
        f"  Mean geo (norm.)   : {geo_norm.mean():.4f}",
        f"  Mean sem (norm.)   : {sem_norm.mean():.4f}",
        f"",
        f"── Bias–structure Spearman r (geo threshold {GEO_THRESHOLD:.0%}) ──",
        f"  Geo heads above threshold : {n_geo_heads} / {len(layer_names) * num_heads}",
        f"  Mean Spearman r    : {corr_mean:.4f}",
        f"  Std  Spearman r    : {corr_std:.4f}",
        f"  Best head          : layer {best_layer}, head {best_head}  (r = {best_r:.4f})",
    ]

    summary_path = sp("summary")
    # Strip .png added by sp() — we want .txt
    summary_path = summary_path.replace(".png", ".txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"  Summary → {summary_path}")

    print(f"  All figures saved to {out_dir}")
    return sem_line_ratio, sem_line_raw, geo_line_ratio, geo_line_raw


# ======================================================================
# Main loop — function boundary guarantees full cleanup
# ======================================================================
sem_lines_ratio = {}
geo_lines_ratio = {}
sem_lines_raw   = {}
geo_lines_raw   = {}

for example in EXAMPLES:
    result = process_example(example)

    # Close every figure the plot functions left open
    plt.close("all")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if result is None:
        continue
    sem_lines_ratio[example] = result[0]
    sem_lines_raw[example]   = result[1]
    geo_lines_ratio[example] = result[2]
    geo_lines_raw[example]   = result[3]

# ======================================================================
# Multi-protein overlay
# ======================================================================
print(f"\nGenerating multi-protein overlay …")

fs = FIG_SIZES["overlay"]
fig, axes = plt.subplots(2, 2, figsize=fs,
                         gridspec_kw={"hspace": 0.45, "wspace": 0.35})

overlay_configs = [
    (axes[0, 0], sem_lines_ratio, "Semantic score (ratio)"),
    (axes[0, 1], geo_lines_ratio, "Geometric score (ratio)"),
    (axes[1, 0], sem_lines_raw,   "Semantic score (raw norm.)"),
    (axes[1, 1], geo_lines_raw,   "Geometric score (raw norm.)"),
]

for ax, lines, title in overlay_configs:
    for name, line in lines.items():
        ax.plot(line, color=EXAMPLE_COLORS.get(name, "gray"), lw=1.8, label=EXAMPLE_LABELS[name])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

fig.suptitle("Geo / Sem routing across proteins", fontsize=13, weight="bold")
plt.tight_layout()
fig.savefig(f"{GLOBAL_DIR}/multi_protein_overlay.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  Overlay saved to {GLOBAL_DIR}/multi_protein_overlay.png")

print("\nPipeline complete.")
"""
