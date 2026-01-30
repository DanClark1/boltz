"""Boltz - Deep learning models for biomolecular structure prediction.

This module provides programmatic access to Boltz models for use as a library.
Models can be loaded and used with PyTorch hooks for inspection, debugging,
and feature extraction.

Example usage with hooks:

    import boltz

    # Load a model
    model = boltz.load_model("boltz2", device="cuda")

    # Store captured activations
    activations = {}

    # Define a hook to capture intermediate outputs
    def capture_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = tuple(
                    o.detach().cpu() if hasattr(o, 'detach') else o
                    for o in output
                )
            elif isinstance(output, dict):
                activations[name] = {
                    k: v.detach().cpu() if hasattr(v, 'detach') else v
                    for k, v in output.items()
                }
            elif hasattr(output, 'detach'):
                activations[name] = output.detach().cpu()
        return hook

    # Register hooks on modules of interest
    model.pairformer_module.register_forward_hook(capture_hook("pairformer"))
    model.structure_module.register_forward_hook(capture_hook("structure"))
    model.input_embedder.register_forward_hook(capture_hook("input_embedder"))

    # Run prediction - hooks will fire during the forward pass
    results = boltz.predict(
        model,
        "protein.yaml",
        use_msa_server=True,
        diffusion_samples=1,
    )

    # Access captured intermediate representations
    print("Pairformer output shapes:")
    s, z = activations["pairformer"]
    print(f"  Sequence embedding (s): {s.shape}")
    print(f"  Pair embedding (z): {z.shape}")

    # Access prediction results
    print(f"Predicted coordinates: {results[0]['coords'].shape}")
    print(f"Confidence (pLDDT): {results[0]['plddt'].mean():.2f}")

Key submodules available for hooks:
    - input_embedder: Processes input features into embeddings
    - msa_module: Multiple sequence alignment processing
    - pairformer_module: Core pairwise transformer stack (returns s, z tuple)
    - structure_module: AtomDiffusion for coordinate prediction
    - distogram_module: Distance prediction
    - confidence_module: Confidence/reliability predictions (pLDDT, pAE, pDE)
    - rel_pos: Relative position encoder

For Boltz2 only:
    - template_module: Template processing (if use_templates=True)
    - diffusion_conditioning: Diffusion conditioning module
    - affinity_module: Affinity prediction (if affinity_prediction=True)
"""
from importlib.metadata import PackageNotFoundError, version

try:  # noqa: SIM105
    __version__ = version("boltz")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

# Lazy imports - these are done inside functions to avoid import errors
# when dependencies aren't installed yet
Boltz1 = None
Boltz2 = None
Manifest = None
Record = None


def _ensure_imports():
    """Lazily import model classes when first needed."""
    global Boltz1, Boltz2, Manifest, Record
    if Boltz1 is None:
        from boltz.model.models.boltz1 import Boltz1 as _Boltz1
        from boltz.model.models.boltz2 import Boltz2 as _Boltz2
        from boltz.data.types import Manifest as _Manifest, Record as _Record
        Boltz1 = _Boltz1
        Boltz2 = _Boltz2
        Manifest = _Manifest
        Record = _Record


__all__ = [
    "Boltz1",
    "Boltz2",
    "Manifest",
    "Record",
    "load_model",
    "predict",
    "__version__",
]


def load_model(
    model_name: str = "boltz2",
    checkpoint: str = None,
    device: str = "cpu",
    use_kernels: bool = True,
    cache_dir: str = None,
    **kwargs,
):
    """Load a Boltz model for programmatic use.

    This function loads a Boltz model and returns it directly, allowing you to:
    - Register PyTorch hooks on any submodule
    - Access intermediate representations
    - Customize inference behavior

    Parameters
    ----------
    model_name : str, optional
        Model to load: "boltz1" or "boltz2". Default is "boltz2".

    Note: This function requires pytorch_lightning and other dependencies to be installed.
    checkpoint : str, optional
        Path to a custom checkpoint file. If None, downloads the default weights.
    device : str, optional
        Device to load the model on ("cpu", "cuda", "cuda:0", etc.). Default is "cpu".
    use_kernels : bool, optional
        Whether to use optimized CUDA kernels (requires GPU with compute >= 8.0).
        Default is True.
    cache_dir : str, optional
        Directory to cache downloaded weights. Defaults to ~/.boltz or $BOLTZ_CACHE.
    **kwargs
        Additional arguments passed to the model's load_from_checkpoint method.
        Common options include:
        - recycling_steps: int (default 3)
        - sampling_steps: int (default 200)
        - diffusion_samples: int (default 1)

    Returns
    -------
    Boltz1 or Boltz2
        The loaded model in eval mode. The model is a PyTorch Lightning module
        with all submodules accessible for hook registration.

    Examples
    --------
    Basic usage:

        >>> import boltz
        >>> model = boltz.load_model("boltz2")
        >>> model.eval()

    With custom checkpoint:

        >>> model = boltz.load_model("boltz2", checkpoint="/path/to/weights.ckpt")

    Register hooks for feature extraction:

        >>> embeddings = {}
        >>> def capture_embedding(name):
        ...     def hook(module, input, output):
        ...         if isinstance(output, tuple):
        ...             embeddings[name] = output[0].detach()
        ...         else:
        ...             embeddings[name] = output.detach()
        ...     return hook
        >>>
        >>> model.pairformer_module.register_forward_hook(capture_embedding("pairformer"))
        >>> model.input_embedder.register_forward_hook(capture_embedding("input"))

    Available submodules for Boltz1:
        - input_embedder
        - rel_pos
        - msa_module (if no_msa=False)
        - pairformer_module
        - structure_module
        - distogram_module
        - confidence_module (if confidence_prediction=True)

    Additional submodules for Boltz2:
        - template_module (if use_templates=True)
        - diffusion_conditioning
        - contact_conditioning
        - affinity_module (if affinity_prediction=True)
        - bfactor_module (if predict_bfactor=True)
    """
    import os
    import urllib.request
    from dataclasses import asdict, dataclass
    from pathlib import Path

    import torch

    # Ensure model classes are imported
    _ensure_imports()

    # Determine cache directory
    if cache_dir is None:
        cache_dir = os.environ.get("BOLTZ_CACHE", os.path.expanduser("~/.boltz"))
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # URLs for model weights
    BOLTZ1_URLS = [
        "https://model-gateway.boltz.bio/boltz1_conf.ckpt",
        "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt",
    ]
    BOLTZ2_URLS = [
        "https://model-gateway.boltz.bio/boltz2_conf.ckpt",
        "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt",
    ]

    # Determine model class and checkpoint
    if model_name.lower() == "boltz2":
        model_cls = Boltz2
        default_ckpt = cache / "boltz2_conf.ckpt"
        urls = BOLTZ2_URLS
    elif model_name.lower() == "boltz1":
        model_cls = Boltz1
        default_ckpt = cache / "boltz1_conf.ckpt"
        urls = BOLTZ1_URLS
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'boltz1' or 'boltz2'.")

    # Determine checkpoint path
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    else:
        ckpt_path = default_ckpt
        # Download if needed
        if not ckpt_path.exists():
            print(f"Downloading {model_name} weights to {ckpt_path}...")
            for i, url in enumerate(urls):
                try:
                    urllib.request.urlretrieve(url, str(ckpt_path))
                    break
                except Exception as e:
                    if i == len(urls) - 1:
                        raise RuntimeError(
                            f"Failed to download model from all URLs. Last error: {e}"
                        ) from e
                    continue

    # Default prediction arguments
    predict_args = {
        "recycling_steps": kwargs.pop("recycling_steps", 3),
        "sampling_steps": kwargs.pop("sampling_steps", 200),
        "diffusion_samples": kwargs.pop("diffusion_samples", 1),
        "max_parallel_samples": kwargs.pop("max_parallel_samples", 1),
        "write_confidence_summary": kwargs.pop("write_confidence_summary", True),
        "write_full_pae": kwargs.pop("write_full_pae", True),
        "write_full_pde": kwargs.pop("write_full_pde", False),
    }

    # Default diffusion parameters - different for Boltz1 vs Boltz2
    if model_name.lower() == "boltz2":
        @dataclass
        class DiffusionParams:
            gamma_0: float = 0.8
            gamma_min: float = 1.0
            noise_scale: float = 1.003
            rho: float = 7
            step_scale: float = 1.5
            sigma_min: float = 0.0001
            sigma_max: float = 160.0
            sigma_data: float = 16.0
            P_mean: float = -1.2
            P_std: float = 1.5
            coordinate_augmentation: bool = True
            alignment_reverse_diff: bool = True
            synchronize_sigmas: bool = True
    else:
        @dataclass
        class DiffusionParams:
            gamma_0: float = 0.605
            gamma_min: float = 1.107
            noise_scale: float = 0.901
            rho: float = 8
            step_scale: float = 1.638
            sigma_min: float = 0.0004
            sigma_max: float = 160.0
            sigma_data: float = 16.0
            P_mean: float = -1.2
            P_std: float = 1.5
            coordinate_augmentation: bool = True
            alignment_reverse_diff: bool = True
            synchronize_sigmas: bool = True
            use_inference_model_cache: bool = True

    @dataclass
    class PairformerArgs:
        num_blocks: int = 64 if model_name.lower() == "boltz2" else 48
        num_heads: int = 16
        dropout: float = 0.0
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        v2: bool = model_name.lower() == "boltz2"

    @dataclass
    class MSAModuleArgs:
        msa_s: int = 64
        msa_blocks: int = 4
        msa_dropout: float = 0.0
        z_dropout: float = 0.0
        use_paired_feature: bool = False  # Set below based on model
        pairwise_head_width: int = 32
        pairwise_num_heads: int = 4
        activation_checkpointing: bool = False
        offload_to_cpu: bool = False
        subsample_msa: bool = False
        num_subsampled_msa: int = 1024

    @dataclass
    class SteeringArgs:
        fk_steering: bool = False
        num_particles: int = 3
        fk_lambda: float = 4.0
        fk_resampling_interval: int = 3
        physical_guidance_update: bool = False
        contact_guidance_update: bool = True
        num_gd_steps: int = 20

    diffusion_params = DiffusionParams()
    pairformer_args = PairformerArgs()
    msa_args = MSAModuleArgs(use_paired_feature=(model_name.lower() == "boltz2"))
    steering_args = SteeringArgs()

    # Load model
    model = model_cls.load_from_checkpoint(
        str(ckpt_path),
        strict=True,
        predict_args=predict_args,
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=use_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
        **kwargs,
    )
    model.eval()

    return model


def predict(
    model,
    input_path: str,
    out_dir: str = None,
    use_msa_server: bool = False,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    device: str = None,
    cache_dir: str = None,
    num_workers: int = 4,
):
    """Run structure prediction with a loaded model.

    This function runs inference and allows PyTorch hooks to be triggered
    during the forward pass. Register hooks on the model before calling this.

    Parameters
    ----------
    model : Boltz1 or Boltz2
        A loaded Boltz model (from `load_model()`).
    input_path : str
        Path to a YAML input file or directory of YAML files.
    out_dir : str, optional
        Output directory for predictions. Defaults to current directory.
    use_msa_server : bool, optional
        Whether to use the MSA server for automatic MSA generation.
    recycling_steps : int, optional
        Number of recycling steps. Default is 3.
    sampling_steps : int, optional
        Number of diffusion sampling steps. Default is 200.
    diffusion_samples : int, optional
        Number of structure samples to generate. Default is 1.
    device : str, optional
        Device to run on. Defaults to model's current device.
    cache_dir : str, optional
        Cache directory for downloaded data. Defaults to ~/.boltz.
    num_workers : int, optional
        Number of data loading workers. Default is 4.

    Returns
    -------
    list[dict]
        List of prediction results, one dict per input. Each dict contains:
        - "coords": Predicted atom coordinates [diffusion_samples, num_atoms, 3]
        - "plddt": Per-residue confidence scores
        - "pae": Predicted aligned error matrix (if available)
        - "ptm": Predicted TM score
        - "iptm": Interface predicted TM score
        - "confidence_score": Overall confidence score
        - "s": Sequence embeddings from trunk
        - "z": Pair embeddings from trunk

    Examples
    --------
    Basic prediction with hooks:

        >>> import boltz
        >>>
        >>> # Load model
        >>> model = boltz.load_model("boltz2", device="cuda")
        >>>
        >>> # Set up hook to capture embeddings
        >>> embeddings = {}
        >>> def capture_hook(name):
        ...     def hook(module, input, output):
        ...         if isinstance(output, tuple):
        ...             embeddings[name] = (output[0].detach().cpu(), output[1].detach().cpu())
        ...         elif isinstance(output, dict):
        ...             embeddings[name] = {k: v.detach().cpu() if hasattr(v, 'detach') else v
        ...                                 for k, v in output.items()}
        ...         elif hasattr(output, 'detach'):
        ...             embeddings[name] = output.detach().cpu()
        ...     return hook
        >>>
        >>> # Register hooks
        >>> model.pairformer_module.register_forward_hook(capture_hook("pairformer"))
        >>> model.input_embedder.register_forward_hook(capture_hook("input_embedder"))
        >>>
        >>> # Run prediction - hooks will fire during forward pass
        >>> results = boltz.predict(model, "protein.yaml", use_msa_server=True)
        >>>
        >>> # Access captured embeddings
        >>> print(embeddings["pairformer"][0].shape)  # sequence embeddings
        >>> print(embeddings["pairformer"][1].shape)  # pair embeddings
    """
    import os
    from pathlib import Path

    import torch

    # Ensure model classes are imported
    _ensure_imports()

    # Determine paths
    input_path = Path(input_path)
    if out_dir is None:
        out_dir = Path.cwd() / input_path.stem
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        cache_dir = os.environ.get("BOLTZ_CACHE", os.path.expanduser("~/.boltz"))
    cache = Path(cache_dir)

    # Determine model type
    is_boltz2 = isinstance(model, Boltz2)

    # Collect input files
    if input_path.is_dir():
        input_files = list(input_path.glob("*.yaml")) + list(input_path.glob("*.yml"))
        input_files += list(input_path.glob("*.fasta")) + list(input_path.glob("*.fa"))
    else:
        input_files = [input_path]

    if not input_files:
        raise ValueError(f"No input files found at {input_path}")

    # Use the existing process_inputs function from main.py
    from boltz.main import process_inputs, download_boltz1, download_boltz2

    # Ensure required data is downloaded
    if is_boltz2:
        download_boltz2(cache)
        ccd_path = None
        mol_dir = cache / "mols"
    else:
        download_boltz1(cache)
        ccd_path = cache / "ccd.pkl"
        mol_dir = cache / "mols" if (cache / "mols").exists() else cache

    # Process inputs using the existing infrastructure
    process_inputs(
        data=input_files,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        use_msa_server=use_msa_server,
        boltz2=is_boltz2,
    )

    # Load the manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Set up data module based on model type
    if is_boltz2:
        from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=out_dir / "processed" / "structures",
            msa_dir=out_dir / "processed" / "msa",
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=out_dir / "processed" / "constraints",
            template_dir=out_dir / "processed" / "templates",
            extra_mols_dir=out_dir / "processed" / "mols",
        )
    else:
        from boltz.data.module.inference import BoltzInferenceDataModule
        data_module = BoltzInferenceDataModule(
            manifest=manifest,
            target_dir=out_dir / "processed" / "structures",
            msa_dir=out_dir / "processed" / "msa",
            num_workers=num_workers,
            constraints_dir=out_dir / "processed" / "constraints",
        )

    # Set up data module
    data_module.setup(stage="predict")
    dataloader = data_module.predict_dataloader()

    # Determine device
    if device is not None:
        model_device = torch.device(device)
    elif next(model.parameters()).is_cuda:
        model_device = next(model.parameters()).device
    else:
        model_device = torch.device("cpu")

    model = model.to(model_device)

    # Run inference
    all_results = []
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Run forward pass - hooks will fire here
            output = model(
                batch,
                recycling_steps=recycling_steps,
                num_sampling_steps=sampling_steps,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=True,
            )

            # Collect results
            result = {
                "coords": output["sample_atom_coords"].cpu(),
                "s": output["s"].cpu(),
                "z": output["z"].cpu(),
                "pdistogram": output["pdistogram"].cpu(),
            }

            # Add confidence outputs if available
            if "plddt" in output:
                result["plddt"] = output["plddt"].cpu()
            if "pae" in output:
                result["pae"] = output["pae"].cpu()
            if "pde" in output:
                result["pde"] = output["pde"].cpu()
            if "ptm" in output:
                result["ptm"] = output["ptm"].cpu()
            if "iptm" in output:
                result["iptm"] = output["iptm"].cpu()
            if "complex_plddt" in output:
                result["complex_plddt"] = output["complex_plddt"].cpu()

            all_results.append(result)

    return all_results
