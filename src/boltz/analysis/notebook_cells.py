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
"""
