#!/usr/bin/env python3
"""
Analyze Rank-1 LoRA using Gemma Scope SAEs

This script interprets what the R1 LoRA is doing by comparing:
1. lora_B to residual stream SAE decoder columns (which features promoted/suppressed)
2. lora_A to transcoder encoder columns (what input patterns trigger the LoRA)

For a down_proj LoRA on Gemma 3 27B:
- down_proj: (18432 intermediate) -> (4608 hidden)
- lora_A: (1, 18432) - receives MLP intermediate activations
- lora_B: (4608, 1) - outputs to residual stream

The LoRA update is: ΔW = (alpha/r) * B @ A
So the full effect is: (alpha/r) * B @ (A @ intermediate_acts)
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import argparse
import json


@dataclass
class LoRAWeights:
    """Container for LoRA weights and metadata."""
    lora_A: torch.Tensor  # (r, intermediate_dim) = (1, 18432)
    lora_B: torch.Tensor  # (hidden_dim, r) = (4608, 1)
    alpha: float
    rank: int
    layer: int
    target_module: str

    @property
    def effective_B(self) -> torch.Tensor:
        """B vector scaled by alpha/r."""
        return (self.alpha / self.rank) * self.lora_B.squeeze()

    @property
    def effective_A(self) -> torch.Tensor:
        """A vector (no scaling needed for comparison)."""
        return self.lora_A.squeeze()


class JumpReLUSAE(torch.nn.Module):
    """JumpReLU SAE from Gemma Scope tutorial."""
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.w_enc = torch.nn.Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = torch.nn.Parameter(torch.zeros(d_sae, d_in))
        self.threshold = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.w_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * F.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_lora_weights(adapter_path: str) -> LoRAWeights:
    """Load LoRA weights from a PEFT adapter directory."""
    adapter_path = Path(adapter_path)

    # Load config
    config_path = adapter_path / "adapter_config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load weights
    weights_path = adapter_path / "adapter_model.safetensors"
    weights = load_file(str(weights_path))

    # Extract layer number from config
    layer = config["layers_to_transform"][0]
    target_module = config["target_modules"][0]

    # Find the weight keys - try different naming conventions
    # PEFT can use different prefixes depending on model architecture
    possible_prefixes = [
        f"base_model.model.model.language_model.layers.{layer}.mlp.{target_module}",  # Gemma 3
        f"base_model.model.model.layers.{layer}.mlp.{target_module}",  # Standard
        f"model.layers.{layer}.mlp.{target_module}",  # Minimal
    ]

    lora_A = None
    lora_B = None

    for prefix in possible_prefixes:
        lora_A_key = f"{prefix}.lora_A.weight"
        lora_B_key = f"{prefix}.lora_B.weight"
        if lora_A_key in weights:
            lora_A = weights[lora_A_key]
            lora_B = weights[lora_B_key]
            print(f"  Found weights with prefix: {prefix}")
            break

    if lora_A is None:
        # Fall back to searching all keys
        print("  Searching for LoRA weights in all keys...")
        for key in weights.keys():
            if "lora_A" in key:
                lora_A = weights[key]
                print(f"  Found lora_A: {key}")
            if "lora_B" in key:
                lora_B = weights[key]
                print(f"  Found lora_B: {key}")

    if lora_A is None or lora_B is None:
        raise ValueError(f"Could not find LoRA weights. Available keys: {list(weights.keys())}")

    return LoRAWeights(
        lora_A=lora_A,
        lora_B=lora_B,
        alpha=config["lora_alpha"],
        rank=config["r"],
        layer=layer,
        target_module=target_module,
    )


def load_gemma_scope_sae(
    model_size: str = "27b",
    category: str = "resid_post",
    layer: int = 20,
    width: str = "16k",
    l0: str = "medium",
    instruction_tuned: bool = True,
    device: str = "cuda",
    use_all_layers: bool = False,
) -> JumpReLUSAE:
    """
    Load a Gemma Scope SAE from HuggingFace.

    Args:
        model_size: Model size (e.g., "27b", "1b")
        category: SAE category ("resid_post", "mlp_out", "attn_out")
        layer: Target layer
        width: SAE width ("16k", "65k", "262k", "1m")
        l0: Sparsity level ("small", "medium", "big")
        instruction_tuned: Whether to use IT or PT variant
        device: Device to load to
        use_all_layers: If True, use *_all variants which have every layer
                        but fewer width options. Required for non-standard layers.

    Note:
        Standard category (resid_post, mlp_out, attn_out) only have SAEs at
        4 layers: 25%, 50%, 65%, 85% of model depth.
        For 27B (46 layers): approximately layers 11, 23, 30, 39

        For arbitrary layers, use resid_post_all (with use_all_layers=True)
        which has every layer but only 16k/65k widths.
    """
    variant = "it" if instruction_tuned else "pt"
    repo_id = f"google/gemma-scope-2-{model_size}-{variant}"

    # Use *_all category for non-standard layers
    if use_all_layers:
        actual_category = f"{category}_all"
    else:
        actual_category = category

    filename = f"{actual_category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"

    print(f"Loading SAE from {repo_id}/{filename}")

    try:
        path_to_params = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as e:
        # If standard category fails, try _all variant
        if not use_all_layers:
            print(f"  Standard SAE not found at layer {layer}, trying {category}_all...")
            actual_category = f"{category}_all"
            filename = f"{actual_category}/layer_{layer}_width_{width}_l0_{l0}/params.safetensors"
            print(f"  Trying: {repo_id}/{filename}")
            path_to_params = hf_hub_download(repo_id=repo_id, filename=filename)
        else:
            raise e

    params = load_file(path_to_params)

    d_model, d_sae = params["w_enc"].shape
    print(f"  SAE dimensions: d_model={d_model}, d_sae={d_sae}")

    sae = JumpReLUSAE(d_model, d_sae)
    sae.load_state_dict(params)
    return sae.to(device)


def compute_cosine_similarities(
    vector: torch.Tensor,
    matrix: torch.Tensor,
    normalize_vector: bool = True,
) -> torch.Tensor:
    """
    Compute cosine similarity between a vector and each column of a matrix.

    Args:
        vector: (d,) tensor
        matrix: (d, n) tensor where each column is a feature
        normalize_vector: whether to normalize the input vector

    Returns:
        (n,) tensor of cosine similarities
    """
    if normalize_vector:
        vector = F.normalize(vector.unsqueeze(0), dim=1).squeeze()

    # Normalize each column of the matrix
    matrix_normalized = F.normalize(matrix, dim=0)

    # Compute dot products
    return vector @ matrix_normalized


def analyze_lora_with_sae(
    lora: LoRAWeights,
    sae: JumpReLUSAE,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Analyze which SAE features the LoRA B vector aligns with.

    The B vector (4608,) goes into the residual stream, so we compare
    it with SAE decoder columns (which are also 4608D).
    """
    # Get the effective B vector (scaled by alpha/r)
    lora_b = lora.effective_B.to(sae.w_dec.device).float()

    # SAE decoder: (d_sae, d_model) - each row is a feature's decoder direction
    # We want to compare with columns, so transpose
    decoder = sae.w_dec.T  # (d_model, d_sae)

    print(f"LoRA B shape: {lora_b.shape}")
    print(f"SAE decoder shape: {decoder.shape}")

    # Compute cosine similarities
    similarities = compute_cosine_similarities(lora_b, decoder)

    # Get top positive and negative similarities
    top_pos_vals, top_pos_idx = similarities.topk(top_k)
    top_neg_vals, top_neg_idx = (-similarities).topk(top_k)
    top_neg_vals = -top_neg_vals

    # Also compute projection magnitudes (how much the LoRA projects onto each feature)
    # This is |lora_b| * cos(theta) * |decoder_col|
    decoder_norms = decoder.norm(dim=0)
    projections = similarities * lora_b.norm() * decoder_norms

    top_proj_pos_vals, top_proj_pos_idx = projections.topk(top_k)
    top_proj_neg_vals, top_proj_neg_idx = (-projections).topk(top_k)
    top_proj_neg_vals = -top_proj_neg_vals

    results = {
        "top_positive_features": top_pos_idx.cpu().tolist(),
        "top_positive_cosine": top_pos_vals.cpu().tolist(),
        "top_negative_features": top_neg_idx.cpu().tolist(),
        "top_negative_cosine": top_neg_vals.cpu().tolist(),
        "top_promoted_features": top_proj_pos_idx.cpu().tolist(),
        "top_promoted_projection": top_proj_pos_vals.cpu().tolist(),
        "top_suppressed_features": top_proj_neg_idx.cpu().tolist(),
        "top_suppressed_projection": top_proj_neg_vals.cpu().tolist(),
    }

    # Create summary DataFrames
    df_positive = pd.DataFrame({
        "feature_idx": top_pos_idx.detach().cpu().numpy(),
        "cosine_sim": top_pos_vals.detach().cpu().numpy(),
        "projection": projections[top_pos_idx].detach().cpu().numpy(),
        "decoder_norm": decoder_norms[top_pos_idx].detach().cpu().numpy(),
    })

    df_negative = pd.DataFrame({
        "feature_idx": top_neg_idx.detach().cpu().numpy(),
        "cosine_sim": top_neg_vals.detach().cpu().numpy(),
        "projection": projections[top_neg_idx].detach().cpu().numpy(),
        "decoder_norm": decoder_norms[top_neg_idx].detach().cpu().numpy(),
    })

    return df_positive, df_negative, similarities


def load_sae_example_data(
    model_size: str = "27b",
    category: str = "resid_post",
    layer: int = 20,
    width: str = "16k",
    l0: str = "medium",
    instruction_tuned: bool = True,
    use_all_layers: bool = False,
) -> Optional[dict]:
    """Load example activation data for feature interpretation."""
    variant = "it" if instruction_tuned else "pt"
    repo_id = f"google/gemma-scope-2-{model_size}-{variant}"

    # Use _all category if specified
    actual_category = f"{category}_all" if use_all_layers else category
    filename = f"{actual_category}/layer_{layer}_width_{width}_l0_{l0}/examples.safetensors"

    try:
        print(f"Loading example data from {repo_id}/{filename}")
        path_to_data = hf_hub_download(repo_id=repo_id, filename=filename)
        return load_file(path_to_data)
    except Exception as e:
        print(f"  Could not load example data: {e}")
        # Try without _all suffix as fallback
        if use_all_layers:
            filename_fallback = f"{category}/layer_{layer}_width_{width}_l0_{l0}/examples.safetensors"
            try:
                print(f"  Trying fallback: {filename_fallback}")
                path_to_data = hf_hub_download(repo_id=repo_id, filename=filename_fallback)
                return load_file(path_to_data)
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
        return None


def get_neuronpedia_url(
    model_size: str,
    layer: int,
    feature_idx: int,
    width: str = "16k",
    l0: str = "medium",
    instruction_tuned: bool = True,
    category: str = "resid_post",
) -> str:
    """
    Generate Neuronpedia URL for a feature.

    URL format: https://neuronpedia.org/gemma-scope-2/{model}-{variant}/{sae_id}/{feature_idx}
    Example: https://neuronpedia.org/gemma-scope-2/27b-it/resid_post-layer_20-width_16k-l0_medium/1234
    """
    variant = "it" if instruction_tuned else "pt"
    sae_id = f"{category}-layer_{layer}-width_{width}-l0_{l0}"
    return f"https://neuronpedia.org/gemma-scope-2/{model_size}-{variant}/{sae_id}/{feature_idx}"


def get_feature_info_from_examples(
    example_data: dict,
    feature_idx: int,
    tokenizer=None,
) -> dict:
    """
    Extract interpretable information about a feature from example data.

    Returns dict with:
    - top_tokens: Most common tokens when feature fires (decoded if tokenizer provided)
    - bottom_tokens: Tokens where feature has lowest activation
    - frequency: How often the feature fires
    - top_logits/bottom_logits: What tokens the feature promotes/suppresses
    """
    info = {"feature_idx": feature_idx}

    if example_data is None:
        return info

    # Feature frequency
    if "feature_frequencies" in example_data:
        freq = example_data["feature_frequencies"]
        if feature_idx < len(freq):
            info["frequency"] = float(freq[feature_idx])

    # Top tokens (tokens that commonly appear when feature fires)
    if "top_tokens" in example_data:
        top_toks = example_data["top_tokens"]
        if feature_idx < len(top_toks):
            token_ids = top_toks[feature_idx].tolist()
            if tokenizer is not None:
                info["top_tokens"] = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                info["top_token_ids"] = token_ids

    # Bottom tokens (tokens where feature has lowest activation)
    if "bottom_tokens" in example_data:
        bottom_toks = example_data["bottom_tokens"]
        if feature_idx < len(bottom_toks):
            token_ids = bottom_toks[feature_idx].tolist()
            if tokenizer is not None:
                info["bottom_tokens"] = tokenizer.convert_ids_to_tokens(token_ids)
            else:
                info["bottom_token_ids"] = token_ids

    # Top logits (tokens the feature promotes when active)
    if "top_logits" in example_data and "tokens" in example_data:
        # Note: top_logits shape is (n_features, k) with logit values
        # We need to pair with the actual token predictions
        pass  # Complex to extract without more context

    # Logit effects (what the feature predicts)
    if "logit_effects" in example_data:
        logits = example_data["logit_effects"]
        if feature_idx < len(logits):
            feature_logits = logits[feature_idx]
            # Get indices of non-zero logit effects
            nonzero_mask = feature_logits != 0
            if nonzero_mask.any():
                info["mean_logit_effect"] = float(feature_logits[nonzero_mask].mean())

    return info


def load_tokenizer(model_name: str = "google/gemma-3-27b-it"):
    """Load tokenizer for decoding token IDs."""
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from {model_name}...")
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return None




def load_transcoder(
    model_size: str = "27b",
    layer: int = 20,
    width: str = "65k",
    l0: str = "medium",
    affine: bool = True,
    instruction_tuned: bool = True,
    device: str = "cuda",
) -> Tuple[JumpReLUSAE, dict]:
    """
    Load a Gemma Scope transcoder.

    Transcoders map MLP input -> MLP output, so their encoder
    operates on the same space as the down_proj input (intermediate dim).
    """
    variant = "it" if instruction_tuned else "pt"
    repo_id = f"google/gemma-scope-2-{model_size}-{variant}"
    affine_str = "_affine" if affine else ""
    filename = f"transcoder/layer_{layer}_width_{width}_l0_{l0}{affine_str}/params.safetensors"

    print(f"Loading transcoder from {repo_id}/{filename}")

    try:
        path_to_params = hf_hub_download(repo_id=repo_id, filename=filename)
        params = load_file(path_to_params)

        d_model, d_sae = params["w_enc"].shape
        print(f"  Transcoder dimensions: d_in={d_model}, d_sae={d_sae}")

        # Transcoders have an affine skip connection
        class TranscoderSAE(JumpReLUSAE):
            def __init__(self, d_in, d_sae, has_affine=False):
                super().__init__(d_in, d_sae)
                if has_affine:
                    self.affine_skip = torch.nn.Parameter(torch.zeros(d_in, d_in))
                else:
                    self.affine_skip = None

            def forward(self, x):
                recon = super().forward(x)
                if self.affine_skip is not None:
                    recon = recon + x @ self.affine_skip
                return recon

        has_affine = "affine_skip_connection" in params
        tc = TranscoderSAE(d_model, d_sae, has_affine=has_affine)

        # Load state dict (handle affine skip connection name difference)
        state_dict = {k: v for k, v in params.items()}
        if "affine_skip_connection" in state_dict:
            state_dict["affine_skip"] = state_dict.pop("affine_skip_connection")
        tc.load_state_dict(state_dict, strict=False)

        return tc.to(device), {"d_in": d_model, "d_sae": d_sae}
    except Exception as e:
        print(f"  Could not load transcoder: {e}")
        return None, {}


def analyze_lora_A_with_transcoder(
    lora: LoRAWeights,
    transcoder: JumpReLUSAE,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """
    Analyze which transcoder features the LoRA A vector aligns with.

    The A vector (18432,) receives MLP intermediate activations.
    The transcoder encoder also operates on the MLP input (pre-layernorm resid).

    Note: This is an approximate analysis since:
    - Transcoder input is pre-MLP layernorm (4608 dim)
    - LoRA A receives intermediate activations (18432 dim after gate/up proj)

    We can still look at what transcoder latents the LoRA A might be
    sensitive to by looking at encoder weight correlations.
    """
    # Get the A vector
    lora_a = lora.effective_A.to(transcoder.w_enc.device).float()

    # Transcoder encoder: (d_in, d_sae) where d_in is 4608 (resid dim)
    # This doesn't directly match lora_A which is 18432 dim
    # But we can analyze the decoder side which outputs to 4608 dim

    # The transcoder decoder: (d_sae, d_out) where d_out is 4608
    # Each row is a latent's output direction
    decoder = transcoder.w_dec  # (d_sae, d_out=4608)

    print(f"LoRA A shape: {lora_a.shape}")
    print(f"Transcoder decoder shape: {decoder.shape}")

    # Note: Direct comparison isn't possible since dimensions don't match
    # lora_A: (18432,) - operates on intermediate
    # transcoder: operates on 4608 dim input/output

    print("  Note: Transcoder operates on 4608D space, LoRA A on 18432D intermediate")
    print("  Direct comparison not possible - consider alternative analysis")

    return None, None, None


def print_analysis_results(
    df_positive: pd.DataFrame,
    df_negative: pd.DataFrame,
    lora: LoRAWeights,
    model_size: str = "27b",
    sae_layer: int = 20,
    width: str = "16k",
    l0: str = "medium",
    example_data: Optional[dict] = None,
    tokenizer=None,
    show_urls: bool = True,
    show_bottom_tokens: bool = False,
):
    """Print formatted analysis results with decoded tokens."""
    print("\n" + "=" * 80)
    print(f"LoRA Analysis Results")
    print(f"  Layer: {lora.layer}, Target: {lora.target_module}")
    print(f"  Rank: {lora.rank}, Alpha: {lora.alpha}")
    print(f"  Effective scale: {lora.alpha / lora.rank}")
    print("=" * 80)

    print("\n TOP PROMOTED FEATURES (positive cosine similarity):")
    print("-" * 80)
    if show_bottom_tokens:
        print(f"{'Idx':>7} | {'Cosine':>7} | {'Proj':>8} | {'Freq':>10} | Top Tokens / Bottom Tokens")
    else:
        print(f"{'Idx':>7} | {'Cosine':>7} | {'Proj':>8} | {'Freq':>10} | Top Tokens")
    print("-" * 80)

    for _, row in df_positive.iterrows():
        fidx = int(row["feature_idx"])
        info = get_feature_info_from_examples(example_data, fidx, tokenizer)
        freq_str = f"{info.get('frequency', 0):.2e}" if 'frequency' in info else "N/A"
        top_toks = info.get('top_tokens', info.get('top_token_ids', []))
        tok_str = str(top_toks[:5]) if top_toks else "N/A"
        if show_bottom_tokens:
            bottom_toks = info.get('bottom_tokens', info.get('bottom_token_ids', []))
            bottom_str = str(bottom_toks[:5]) if bottom_toks else "N/A"
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {row['projection']:>8.4f} | {freq_str:>10} | {tok_str}")
            print(f"{'':>7} | {'':>7} | {'':>8} | {'':>10} | Bottom: {bottom_str}")
        else:
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {row['projection']:>8.4f} | {freq_str:>10} | {tok_str}")

    if show_urls and len(df_positive) > 0:
        print("\nNeuronpedia URLs for top 5 promoted features:")
        for _, row in df_positive.head(5).iterrows():
            fidx = int(row["feature_idx"])
            url = get_neuronpedia_url(model_size, sae_layer, fidx, width, l0)
            print(f"  Feature {fidx}: {url}")

    print("\n TOP SUPPRESSED FEATURES (negative cosine similarity):")
    print("-" * 80)
    if show_bottom_tokens:
        print(f"{'Idx':>7} | {'Cosine':>7} | {'Proj':>8} | {'Freq':>10} | Top Tokens / Bottom Tokens")
    else:
        print(f"{'Idx':>7} | {'Cosine':>7} | {'Proj':>8} | {'Freq':>10} | Top Tokens")
    print("-" * 80)

    for _, row in df_negative.iterrows():
        fidx = int(row["feature_idx"])
        info = get_feature_info_from_examples(example_data, fidx, tokenizer)
        freq_str = f"{info.get('frequency', 0):.2e}" if 'frequency' in info else "N/A"
        top_toks = info.get('top_tokens', info.get('top_token_ids', []))
        tok_str = str(top_toks[:5]) if top_toks else "N/A"
        if show_bottom_tokens:
            bottom_toks = info.get('bottom_tokens', info.get('bottom_token_ids', []))
            bottom_str = str(bottom_toks[:5]) if bottom_toks else "N/A"
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {row['projection']:>8.4f} | {freq_str:>10} | {tok_str}")
            print(f"{'':>7} | {'':>7} | {'':>8} | {'':>10} | Bottom: {bottom_str}")
        else:
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {row['projection']:>8.4f} | {freq_str:>10} | {tok_str}")

    if show_urls and len(df_negative) > 0:
        print("\nNeuronpedia URLs for top 5 suppressed features:")
        for _, row in df_negative.head(5).iterrows():
            fidx = int(row["feature_idx"])
            url = get_neuronpedia_url(model_size, sae_layer, fidx, width, l0)
            print(f"  Feature {fidx}: {url}")


def compare_multiple_loras(
    lora_paths: list,
    sae: JumpReLUSAE,
    top_k: int = 10,
) -> pd.DataFrame:
    """Compare multiple LoRAs by their alignment with SAE features."""
    results = []

    for path in lora_paths:
        lora = load_lora_weights(path)
        df_pos, df_neg, all_sims = analyze_lora_with_sae(lora, sae, top_k=top_k)

        # Get top features
        top_features = df_pos["feature_idx"].tolist()[:5]

        results.append({
            "path": Path(path).name,
            "alpha": lora.alpha,
            "rank": lora.rank,
            "max_cos": all_sims.max().item(),
            "min_cos": all_sims.min().item(),
            "std_cos": all_sims.std().item(),
            "top_5_promoted": top_features,
            "top_5_suppressed": df_neg["feature_idx"].tolist()[:5],
        })

    return pd.DataFrame(results)


def load_base_model_down_proj(
    model_name: str = "google/gemma-3-27b-it",
    layer: int = 20,
    device: str = "cpu",
) -> torch.Tensor:
    """Load just the down_proj weights from the base model."""
    from transformers import AutoModelForCausalLM
    import gc

    print(f"Loading down_proj from {model_name} layer {layer}...")

    # Load model with minimal memory footprint
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Keep on CPU to save GPU memory
        low_cpu_mem_usage=True,
    )

    # Extract just the down_proj weight
    # Try different model structures
    try:
        # Gemma 3 structure: model.language_model.layers[i].mlp.down_proj
        down_proj = model.language_model.layers[layer].mlp.down_proj.weight.clone()
        print(f"  Found down_proj via language_model.layers")
    except AttributeError:
        try:
            # Alternative: model.model.layers[i].mlp.down_proj
            down_proj = model.model.layers[layer].mlp.down_proj.weight.clone()
            print(f"  Found down_proj via model.layers")
        except AttributeError:
            # Print model structure to debug
            print(f"  Model type: {type(model)}")
            print(f"  Model attributes: {[a for a in dir(model) if not a.startswith('_')]}")
            if hasattr(model, 'language_model'):
                print(f"  language_model type: {type(model.language_model)}")
                print(f"  language_model attrs: {[a for a in dir(model.language_model) if not a.startswith('_')]}")
            raise

    # Clean up
    del model
    gc.collect()

    print(f"  down_proj shape: {down_proj.shape}")
    return down_proj.to(device)


def analyze_lora_A_with_base_model(
    lora: LoRAWeights,
    down_proj: torch.Tensor,
    sae: JumpReLUSAE,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """
    Analyze what the A pattern normally produces via down_proj.

    Computes W_down @ A^T to get the residual direction that A inputs
    would normally produce, then compares to SAE features.
    """
    # A is (1, intermediate_dim), we want (intermediate_dim, 1)
    A = lora.lora_A.T.to(down_proj.device).float()  # (21504, 1)

    # down_proj weight is (hidden_dim, intermediate_dim) = (5376, 21504)
    W_down = down_proj.float()

    # W_down @ A gives (5376, 1) - what A would produce through down_proj
    normal_output = (W_down @ A).squeeze()  # (5376,)

    print(f"A shape: {lora.lora_A.shape}")
    print(f"W_down shape: {W_down.shape}")
    print(f"W_down @ A^T shape: {normal_output.shape}")
    print(f"|W_down @ A^T|: {normal_output.norm().item():.4f}")

    # Compare to SAE decoder columns
    decoder = sae.w_dec.T.to(normal_output.device)  # (d_model, d_sae)

    similarities = compute_cosine_similarities(normal_output, decoder)

    # Get top positive and negative
    top_pos_vals, top_pos_idx = similarities.topk(top_k)
    top_neg_vals, top_neg_idx = (-similarities).topk(top_k)
    top_neg_vals = -top_neg_vals

    df_positive = pd.DataFrame({
        "feature_idx": top_pos_idx.detach().cpu().numpy(),
        "cosine_sim": top_pos_vals.detach().cpu().numpy(),
    })

    df_negative = pd.DataFrame({
        "feature_idx": top_neg_idx.detach().cpu().numpy(),
        "cosine_sim": top_neg_vals.detach().cpu().numpy(),
    })

    # Also compute alignment between B and W_down @ A^T
    B = lora.effective_B.to(normal_output.device).float()
    B_alignment = F.cosine_similarity(B.unsqueeze(0), normal_output.unsqueeze(0)).item()
    print(f"\nAlignment between B and W_down @ A^T: {B_alignment:.4f}")
    if B_alignment > 0.1:
        print("  -> LoRA AMPLIFIES what the model normally does with A inputs")
    elif B_alignment < -0.1:
        print("  -> LoRA SUPPRESSES what the model normally does with A inputs")
    else:
        print("  -> LoRA REDIRECTS (orthogonal to normal A output)")

    return df_positive, df_negative, similarities, normal_output, B_alignment


def main():
    parser = argparse.ArgumentParser(description="Analyze R1 LoRA with Gemma Scope SAE")
    parser.add_argument(
        "--lora-path",
        type=str,
        default="/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20/2026-01-15_18-10-31",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--sae-layer",
        type=int,
        default=20,
        help="SAE layer to load (should match LoRA layer)",
    )
    parser.add_argument(
        "--sae-width",
        type=str,
        default="16k",
        choices=["16k", "65k", "262k", "1m"],
        help="SAE width",
    )
    parser.add_argument(
        "--sae-l0",
        type=str,
        default="medium",
        choices=["small", "medium", "big"],
        help="SAE L0 (sparsity level)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="27b",
        help="Gemma model size",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top features to show",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (optional)",
    )
    parser.add_argument(
        "--load-examples",
        action="store_true",
        help="Load example activation data for feature interpretation",
    )
    parser.add_argument(
        "--show-bottom-tokens",
        action="store_true",
        help="Also show bottom activating tokens for each feature",
    )
    parser.add_argument(
        "--analyze-A",
        action="store_true",
        help="Also analyze A matrix via W_down @ A^T (requires loading base model)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-3-27b-it",
        help="Base model to load down_proj from (for --analyze-A)",
    )
    parser.add_argument(
        "--use-all-layers",
        action="store_true",
        help="Use *_all SAE variants (available for every layer, fewer widths)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    print("Loading LoRA weights...")
    lora = load_lora_weights(args.lora_path)
    print(f"  Loaded LoRA: layer={lora.layer}, alpha={lora.alpha}, rank={lora.rank}")
    print(f"  lora_A shape: {lora.lora_A.shape}")
    print(f"  lora_B shape: {lora.lora_B.shape}")

    print("\nLoading Gemma Scope SAE...")
    sae = load_gemma_scope_sae(
        model_size=args.model_size,
        category="resid_post",
        layer=args.sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
        instruction_tuned=True,
        device=args.device,
        use_all_layers=args.use_all_layers,
    )

    # Optionally load example data and tokenizer
    example_data = None
    tokenizer = None
    if args.load_examples:
        example_data = load_sae_example_data(
            model_size=args.model_size,
            category="resid_post",
            layer=args.sae_layer,
            width=args.sae_width,
            l0=args.sae_l0,
            instruction_tuned=True,
            use_all_layers=args.use_all_layers,
        )
        # Load tokenizer for decoding token IDs
        base_model = f"google/gemma-3-{args.model_size}-it"
        tokenizer = load_tokenizer(base_model)

    print("\nAnalyzing LoRA-SAE alignment...")
    df_pos, df_neg, all_sims = analyze_lora_with_sae(lora, sae, top_k=args.top_k)

    print_analysis_results(
        df_pos, df_neg, lora,
        model_size=args.model_size,
        sae_layer=args.sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
        example_data=example_data,
        tokenizer=tokenizer,
        show_bottom_tokens=args.show_bottom_tokens,
    )

    # Summary statistics
    print("\n DISTRIBUTION STATISTICS:")
    print(f"  Mean cosine sim: {all_sims.mean().item():.6f}")
    print(f"  Std cosine sim: {all_sims.std().item():.6f}")
    print(f"  Max cosine sim: {all_sims.max().item():.6f}")
    print(f"  Min cosine sim: {all_sims.min().item():.6f}")
    print(f"  Features > 0.1: {(all_sims > 0.1).sum().item()}")
    print(f"  Features < -0.1: {(all_sims < -0.1).sum().item()}")

    # LoRA vector statistics
    lora_b_norm = lora.lora_B.norm().item()
    lora_a_norm = lora.lora_A.norm().item()
    effective_norm = (lora.alpha / lora.rank) * lora_b_norm
    print(f"\n LORA VECTOR STATISTICS:")
    print(f"  |lora_B|: {lora_b_norm:.4f}")
    print(f"  |lora_A|: {lora_a_norm:.4f}")
    print(f"  |effective_B| (scaled): {effective_norm:.4f}")

    # Analyze A matrix if requested
    if args.analyze_A:
        print("\n" + "=" * 80)
        print("ANALYZING A MATRIX (W_down @ A^T)")
        print("=" * 80)

        down_proj = load_base_model_down_proj(
            model_name=args.base_model,
            layer=lora.layer,
            device=args.device,
        )

        df_A_pos, df_A_neg, A_sims, normal_output, B_alignment = analyze_lora_A_with_base_model(
            lora, down_proj, sae, top_k=args.top_k
        )

        print("\n W_down @ A^T - MAX COSINE SIMILARITY FEATURES:")
        print("-" * 80)
        print(f"{'Idx':>7} | {'Cosine':>7} | Top Tokens")
        print("-" * 80)
        for _, row in df_A_pos.head(15).iterrows():
            fidx = int(row["feature_idx"])
            info = get_feature_info_from_examples(example_data, fidx, tokenizer)
            top_toks = info.get('top_tokens', info.get('top_token_ids', []))
            tok_str = str(top_toks[:5]) if top_toks else "N/A"
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {tok_str}")

        print("\n W_down @ A^T - MIN COSINE SIMILARITY FEATURES:")
        print("-" * 80)
        print(f"{'Idx':>7} | {'Cosine':>7} | Top Tokens")
        print("-" * 80)
        for _, row in df_A_neg.head(15).iterrows():
            fidx = int(row["feature_idx"])
            info = get_feature_info_from_examples(example_data, fidx, tokenizer)
            top_toks = info.get('top_tokens', info.get('top_token_ids', []))
            tok_str = str(top_toks[:5]) if top_toks else "N/A"
            print(f"{fidx:>7} | {row['cosine_sim']:>7.4f} | {tok_str}")

        print(f"\n A MATRIX STATISTICS:")
        print(f"  Mean cosine sim: {A_sims.mean().item():.6f}")
        print(f"  Std cosine sim: {A_sims.std().item():.6f}")
        print(f"  Max cosine sim: {A_sims.max().item():.6f}")
        print(f"  Min cosine sim: {A_sims.min().item():.6f}")
        print(f"  B alignment with W_down @ A^T: {B_alignment:.4f}")

        # Save A matrix analysis results
        if args.output:
            A_summary = {
                "analysis_type": "A_matrix",
                "description": "W_down @ A^T compared to SAE decoder",
                "lora_path": args.lora_path,
                "lora_layer": lora.layer,
                "sae_layer": args.sae_layer,
                "sae_width": args.sae_width,
                "sae_l0": args.sae_l0,
                "max_cos": A_sims.max().item(),
                "min_cos": A_sims.min().item(),
                "mean_cos": A_sims.mean().item(),
                "std_cos": A_sims.std().item(),
                "B_alignment": B_alignment,
                "top_max_cos": df_A_pos["feature_idx"].tolist(),
                "top_min_cos": df_A_neg["feature_idx"].tolist(),
            }
            A_output_path = str(Path(args.output)) + ".A_matrix.json"
            with open(A_output_path, "w") as f:
                json.dump(A_summary, f, indent=2)
            print(f"\nA matrix results saved to {A_output_path}")

    if args.output:
        output_path = Path(args.output)
        df_pos.to_csv(str(output_path) + ".positive.csv", index=False)
        df_neg.to_csv(str(output_path) + ".negative.csv", index=False)

        # Save full similarities
        np.save(
            str(output_path) + ".all_sims.npy",
            all_sims.detach().cpu().numpy()
        )

        # Save summary
        summary = {
            "lora_path": args.lora_path,
            "lora_layer": lora.layer,
            "lora_alpha": lora.alpha,
            "lora_rank": lora.rank,
            "sae_layer": args.sae_layer,
            "sae_width": args.sae_width,
            "sae_l0": args.sae_l0,
            "max_cos": all_sims.max().item(),
            "min_cos": all_sims.min().item(),
            "mean_cos": all_sims.mean().item(),
            "std_cos": all_sims.std().item(),
            "lora_b_norm": lora_b_norm,
            "lora_a_norm": lora_a_norm,
            "top_promoted": df_pos["feature_idx"].tolist(),
            "top_suppressed": df_neg["feature_idx"].tolist(),
        }
        with open(str(output_path) + ".summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {output_path}.*")

    return df_pos, df_neg, all_sims


if __name__ == "__main__":
    main()
