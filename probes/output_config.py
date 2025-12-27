"""
Centralized output path configuration for research-tools.

This module provides a single source of truth for all output directories,
making it easy to update paths and maintain consistency across scripts.
"""

from pathlib import Path
from typing import Dict, Optional

# Repository root - all paths are relative to this
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Main output directory
OUTPUTS_ROOT = REPO_ROOT / "outputs"

# ============================================================================
# OUTPUT DIRECTORY STRUCTURE
# ============================================================================

class OutputPaths:
    """Central registry of all output paths."""

    # Root directories
    ROOT = OUTPUTS_ROOT
    DATA = OUTPUTS_ROOT / "data"
    DIM_REDUCTION = OUTPUTS_ROOT / "dimensionality_reduction"
    PROBES = OUTPUTS_ROOT / "probes"
    EVALUATIONS = OUTPUTS_ROOT / "evaluations"
    INTERPRETATIONS = OUTPUTS_ROOT / "interpretations"
    VISUALIZATIONS = OUTPUTS_ROOT / "visualizations"

    # Data paths
    class Data:
        """Intermediate activation data paths."""
        ROOT = OUTPUTS_ROOT / "data"
        ACTIVATIONS = ROOT / "activations"
        ACTIVATIONS_REGIONAL = ACTIVATIONS / "regional"

        @classmethod
        def activation_file(cls, name: str) -> Path:
            """Get path for an activation file."""
            return cls.ACTIVATIONS / f"{name}.h5"

        @classmethod
        def regional_file(cls, name: str) -> Path:
            """Get path for a regional activation file."""
            return cls.ACTIVATIONS_REGIONAL / f"{name}.h5"

    # Dimensionality reduction paths
    class DimReduction:
        """cPCA and Ratio PCA output paths."""
        ROOT = OUTPUTS_ROOT / "dimensionality_reduction"

        # cPCA paths
        CPCA = ROOT / "cpca"
        CPCA_TIER_BASED = CPCA / "tier_based"
        CPCA_CONVERSATION = CPCA / "conversation_based"
        CPCA_CONVERSATION_GLOBAL = CPCA_CONVERSATION / "global"
        CPCA_CONVERSATION_REGIONAL = CPCA_CONVERSATION / "regional"
        CPCA_COMBINED = CPCA / "combined_ranges"

        # Ratio PCA paths
        RATIO_PCA = ROOT / "ratio_pca"
        RATIO_PCA_TIER = RATIO_PCA / "tier_based"
        RATIO_PCA_K5 = RATIO_PCA_TIER / "k5"
        RATIO_PCA_K10 = RATIO_PCA_TIER / "k10"
        RATIO_PCA_K30 = RATIO_PCA_TIER / "k30"
        RATIO_PCA_K_TUNING = RATIO_PCA_TIER / "k_tuning"

        # Comparisons
        COMPARISONS = ROOT / "comparisons"
        COMPARISON_PLOTS = COMPARISONS / "method_comparison_plots"

        @classmethod
        def cpca_tier_based(cls, model_name: str, alpha: Optional[float] = None) -> Path:
            """Get cPCA output path for tier-based data."""
            base_dir = cls.CPCA_TIER_BASED
            if alpha is not None:
                base_dir = base_dir / "alpha_sweep" / f"alpha_{alpha}"
            # Create model-specific subdirectory (e.g., google/)
            model_vendor = model_name.split("/")[0] if "/" in model_name else "default"
            return base_dir / model_vendor

        @classmethod
        def cpca_conversation_global(cls, model_name: str) -> Path:
            """Get cPCA output path for conversation global data."""
            model_vendor = model_name.split("/")[0] if "/" in model_name else "default"
            return cls.CPCA_CONVERSATION_GLOBAL / model_vendor

        @classmethod
        def cpca_conversation_regional(cls, region_name: str, model_name: str) -> Path:
            """Get cPCA output path for conversation regional data."""
            model_vendor = model_name.split("/")[0] if "/" in model_name else "default"
            return cls.CPCA_CONVERSATION_REGIONAL / region_name / model_vendor

        @classmethod
        def ratio_pca_k(cls, k: int) -> Path:
            """Get Ratio PCA output path for given k."""
            return cls.RATIO_PCA_TIER / f"k{k}"

    # Probe paths
    class Probes:
        """Trained probe model paths."""
        ROOT = OUTPUTS_ROOT / "probes"
        EMOTION_PROBES = ROOT / "emotion_probes"

        # Text-based emotion probes
        TEXT_BASED = EMOTION_PROBES / "text_based"
        TEXT_RAW = TEXT_BASED / "raw"
        TEXT_CPCA = TEXT_BASED / "cpca"
        TEXT_CPCA_TOP3 = TEXT_CPCA / "top3"
        TEXT_CPCA_TOP5 = TEXT_CPCA / "top5"
        TEXT_CPCA_TOP10 = TEXT_CPCA / "top10"
        TEXT_CPCA_TOP20 = TEXT_CPCA / "top20"
        TEXT_REGULARIZATION = TEXT_BASED / "regularization"
        TEXT_L1 = TEXT_REGULARIZATION / "l1"
        TEXT_HIGH_ALPHA = TEXT_REGULARIZATION / "high_alpha_cpca"
        TEXT_MULTISEED = TEXT_BASED / "multiseed"

        # Conversation-based emotion probes
        CONVERSATION_BASED = EMOTION_PROBES / "conversation_based"
        CONVERSATION_STANDARD = CONVERSATION_BASED / "standard"
        CONVERSATION_ORTHOGONAL = CONVERSATION_BASED / "orthogonal"

        # Manifests
        MANIFESTS = ROOT / "manifests"

        @classmethod
        def text_cpca_topk(cls, k: int) -> Path:
            """Get probe path for top-k cPCA components."""
            return cls.TEXT_CPCA / f"top{k}"

        @classmethod
        def conversation_orthogonal(cls, ortho_weight: float) -> Path:
            """Get path for orthogonal probes with given weight."""
            return cls.CONVERSATION_ORTHOGONAL / f"ortho_{ortho_weight}"

        @classmethod
        def multiseed(cls, seed: int) -> Path:
            """Get path for multi-seed probe with given seed."""
            return cls.TEXT_MULTISEED / f"seed_{seed}"

    # Evaluation paths
    class Evaluations:
        """Evaluation result paths."""
        ROOT = OUTPUTS_ROOT / "evaluations"

        # Conversation evaluation
        CONVERSATION_EVAL = ROOT / "conversation_eval"
        CONVERSATION_TEXT_BASED = CONVERSATION_EVAL / "text_based_probes"
        CONVERSATION_TEXT_RAW = CONVERSATION_TEXT_BASED / "raw"
        CONVERSATION_TEXT_CPCA = CONVERSATION_TEXT_BASED / "cpca_variants"
        CONVERSATION_TEXT_MULTISEED = CONVERSATION_TEXT_BASED / "multiseed"
        CONVERSATION_BASED_PROBES = CONVERSATION_EVAL / "conversation_based_probes"
        CONVERSATION_PLOTS = CONVERSATION_EVAL / "plots"

        # Emo lens experiments
        EMO_LENS = ROOT / "emo_lens_experiments"
        EMO_LENS_SINGLE = EMO_LENS / "single_layer"
        EMO_LENS_MULTI = EMO_LENS / "multilayer"

        # Comparative analysis
        COMPARATIVE = ROOT / "comparative_analysis"
        COMPARATIVE_PCA = COMPARATIVE / "pca_method_comparison"
        COMPARATIVE_REG = COMPARATIVE / "regularization_comparison"
        COMPARATIVE_AUTOINTERP = COMPARATIVE / "autointerp_comparison"

        @classmethod
        def conversation_eval_variant(cls, variant: str) -> Path:
            """Get evaluation path for a specific probe variant."""
            if variant in ["raw", "no_cpca"]:
                return cls.CONVERSATION_TEXT_RAW
            elif "top" in variant:
                return cls.CONVERSATION_TEXT_CPCA / variant
            elif "multiseed" in variant:
                return cls.CONVERSATION_TEXT_MULTISEED
            else:
                return cls.CONVERSATION_TEXT_BASED / variant

        @classmethod
        def emo_lens_experiment(cls, question_module: str, multilayer: bool = False) -> Path:
            """Get emo lens experiment output path."""
            base = cls.EMO_LENS_MULTI if multilayer else cls.EMO_LENS_SINGLE
            return base / question_module

    # Interpretation paths
    class Interpretations:
        """PC interpretation and analysis paths."""
        ROOT = OUTPUTS_ROOT / "interpretations"

        # Auto-interpretation
        AUTOINTERP = ROOT / "autointerp"
        AUTOINTERP_TIER = AUTOINTERP / "tier_based"
        AUTOINTERP_CONVERSATION = AUTOINTERP / "conversation_based"
        AUTOINTERP_RATIO_PCA = AUTOINTERP / "ratio_pca"

        # PC analysis
        PC_ANALYSIS = ROOT / "pc_analysis"
        PC_WEIGHTS = PC_ANALYSIS / "weight_distributions"
        PC_MEANINGS = PC_ANALYSIS / "component_meanings"
        PC_SEMANTIC = PC_ANALYSIS / "semantic_alignment"

        @classmethod
        def autointerp_tier(cls, tier_name: str) -> Path:
            """Get autointerp path for a specific tier."""
            return cls.AUTOINTERP_TIER / tier_name

        @classmethod
        def autointerp_conversation_regional(cls, region_name: str) -> Path:
            """Get autointerp path for a conversation region."""
            return cls.AUTOINTERP_CONVERSATION / f"regional_{region_name}"

    # Visualization paths
    class Visualizations:
        """Plot and figure output paths."""
        ROOT = OUTPUTS_ROOT / "visualizations"

        # Probe performance
        PROBE_PERFORMANCE = ROOT / "probe_performance"
        ACCURACY_BY_LAYER = PROBE_PERFORMANCE / "accuracy_by_layer"
        ACCURACY_HEATMAPS = PROBE_PERFORMANCE / "accuracy_heatmaps"
        DIMENSIONALITY_ANALYSIS = PROBE_PERFORMANCE / "dimensionality_analysis"

        # Conversation analysis
        CONVERSATION_ANALYSIS = ROOT / "conversation_analysis"
        USER_VS_ASSISTANT = CONVERSATION_ANALYSIS / "user_vs_assistant"
        ORTHOGONAL_PROBES = CONVERSATION_ANALYSIS / "orthogonal_probes"

        # Component analysis
        COMPONENT_ANALYSIS = ROOT / "component_analysis"

        # Emotion trajectories
        EMOTION_TRAJECTORIES = ROOT / "emotion_trajectories"

        # Method comparisons
        METHOD_COMPARISONS = ROOT / "method_comparisons"


# ============================================================================
# LEGACY PATH MAPPING (for backward compatibility)
# ============================================================================

LEGACY_TO_NEW_PATHS: Dict[str, Path] = {
    # Old results/ paths
    "results/emotion_probes": OutputPaths.Probes.TEXT_RAW,
    "results/emotion_probes_raw": OutputPaths.Probes.TEXT_RAW,
    "results/emotion_probes_top3": OutputPaths.Probes.TEXT_CPCA_TOP3,
    "results/emotion_probes_top5": OutputPaths.Probes.TEXT_CPCA_TOP5,
    "results/emotion_probes_top10": OutputPaths.Probes.TEXT_CPCA_TOP10,
    "results/emotion_probes_top20": OutputPaths.Probes.TEXT_CPCA_TOP20,
    "results/emotion_probes_l1": OutputPaths.Probes.TEXT_L1,
    "results/emotion_probes_high_alpha_cpca": OutputPaths.Probes.TEXT_HIGH_ALPHA,
    "results/emotion_probes_multiseed": OutputPaths.Probes.TEXT_MULTISEED,

    "results/conversation_eval": OutputPaths.Evaluations.CONVERSATION_EVAL,
    "results/emo_lens_probe_experiments": OutputPaths.Evaluations.EMO_LENS_SINGLE,
    "results/emo_lens_probe_experiments_multilayer": OutputPaths.Evaluations.EMO_LENS_MULTI,
    "results/autointerp": OutputPaths.Interpretations.AUTOINTERP_TIER,
    "results/pc_weight_visualizations": OutputPaths.Visualizations.COMPONENT_ANALYSIS,

    # Old probes/results/ paths
    "probes/results/cpca_tier_data": OutputPaths.DimReduction.CPCA_TIER_BASED,
    "probes/results/cpca_tier_data_high_alpha": OutputPaths.DimReduction.CPCA_TIER_BASED / "alpha_sweep" / "alpha_5.0",
    "probes/results/cpca_conversations_global": OutputPaths.DimReduction.CPCA_CONVERSATION_GLOBAL,
    "probes/results/cpca_conversations_regional_user": OutputPaths.DimReduction.CPCA_CONVERSATION_REGIONAL / "user",
    "probes/results/cpca_conversations_regional_asst": OutputPaths.DimReduction.CPCA_CONVERSATION_REGIONAL / "assistant",
    "probes/results/cpca_conversations_regional_special1": OutputPaths.DimReduction.CPCA_CONVERSATION_REGIONAL / "special_tokens_1",
    "probes/results/cpca_conversations_regional_special2": OutputPaths.DimReduction.CPCA_CONVERSATION_REGIONAL / "special_tokens_2",

    "probes/results/conversation_probes": OutputPaths.Probes.CONVERSATION_STANDARD,
    "probes/results/conversation_probes_orthogonal": OutputPaths.Probes.CONVERSATION_ORTHOGONAL,
    "probes/results/autointerp": OutputPaths.Interpretations.AUTOINTERP_TIER,
}


def get_output_path(key: str, default: Optional[Path] = None) -> Path:
    """
    Get output path for a given key, with fallback to legacy path mapping.

    Args:
        key: Path key or legacy path string
        default: Default path if key not found

    Returns:
        Path object for the output location

    Examples:
        >>> get_output_path("results/emotion_probes_top10")
        Path('outputs/probes/emotion_probes/text_based/cpca/top10')

        >>> get_output_path("probes/results/cpca_tier_data")
        Path('outputs/dimensionality_reduction/cpca/tier_based')
    """
    # Try legacy mapping first
    if key in LEGACY_TO_NEW_PATHS:
        return LEGACY_TO_NEW_PATHS[key]

    # Try as direct Path
    path = Path(key)
    if path.is_absolute():
        return path

    # Return default or original path
    return default if default is not None else REPO_ROOT / path


def ensure_output_dir(path: Path) -> Path:
    """
    Ensure output directory exists, creating it if necessary.

    Args:
        path: Output directory path

    Returns:
        The created/verified path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def migrate_legacy_path(old_path: str) -> Path:
    """
    Migrate a legacy path to the new unified structure.

    Args:
        old_path: Old path string (e.g., "results/emotion_probes_top10")

    Returns:
        New unified path

    Raises:
        ValueError: If path cannot be migrated
    """
    if old_path in LEGACY_TO_NEW_PATHS:
        return LEGACY_TO_NEW_PATHS[old_path]

    # Try to infer from pattern
    if old_path.startswith("results/emotion_probes"):
        if "_top" in old_path:
            k = old_path.split("_top")[1].split("/")[0]
            return OutputPaths.Probes.text_cpca_topk(int(k))
        elif "_l1" in old_path:
            return OutputPaths.Probes.TEXT_L1
        elif "_multiseed" in old_path:
            return OutputPaths.Probes.TEXT_MULTISEED
        elif "_raw" in old_path:
            return OutputPaths.Probes.TEXT_RAW
        else:
            return OutputPaths.Probes.TEXT_RAW

    raise ValueError(f"Cannot migrate legacy path: {old_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_probe_output_dir(
    probe_type: str = "text",
    representation: str = "raw",
    n_components: Optional[int] = None,
    ortho_weight: Optional[float] = None,
    seed: Optional[int] = None,
    l1_reg: bool = False,
    high_alpha: bool = False
) -> Path:
    """
    Get appropriate output directory for a probe based on configuration.

    Args:
        probe_type: "text" or "conversation"
        representation: "raw", "cpca", "global_cpca", "regional_cpca"
        n_components: Number of PC components (for cPCA)
        ortho_weight: Orthogonality weight (for conversation probes)
        seed: Random seed (for multiseed experiments)
        l1_reg: Whether using L1 regularization
        high_alpha: Whether using high-alpha cPCA

    Returns:
        Output directory path
    """
    if probe_type == "text":
        if seed is not None:
            return OutputPaths.Probes.multiseed(seed)
        elif l1_reg:
            return OutputPaths.Probes.TEXT_L1
        elif high_alpha:
            return OutputPaths.Probes.TEXT_HIGH_ALPHA
        elif representation == "raw":
            return OutputPaths.Probes.TEXT_RAW
        elif representation == "cpca" and n_components:
            return OutputPaths.Probes.text_cpca_topk(n_components)
        else:
            return OutputPaths.Probes.TEXT_RAW

    elif probe_type == "conversation":
        if ortho_weight is not None:
            return OutputPaths.Probes.conversation_orthogonal(ortho_weight)
        else:
            return OutputPaths.Probes.CONVERSATION_STANDARD

    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")


def get_cpca_output_dir(
    data_type: str = "tier_based",
    model_name: str = "google/gemma-3-27b-it",
    alpha: Optional[float] = None,
    region: Optional[str] = None
) -> Path:
    """
    Get appropriate output directory for cPCA results.

    Args:
        data_type: "tier_based" or "conversation"
        model_name: Model name (e.g., "google/gemma-3-27b-it")
        alpha: Alpha parameter value (for high-alpha variants)
        region: Region name for conversation-based (e.g., "user", "assistant")

    Returns:
        Output directory path
    """
    if data_type == "tier_based":
        return OutputPaths.DimReduction.cpca_tier_based(model_name, alpha)
    elif data_type == "conversation":
        if region:
            return OutputPaths.DimReduction.cpca_conversation_regional(region, model_name)
        else:
            return OutputPaths.DimReduction.cpca_conversation_global(model_name)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


if __name__ == "__main__":
    # Print example paths for verification
    print("Output Configuration Examples:")
    print("=" * 80)

    print("\n📁 Probe Paths:")
    print(f"  Raw text probes: {OutputPaths.Probes.TEXT_RAW}")
    print(f"  Top-10 cPCA probes: {OutputPaths.Probes.TEXT_CPCA_TOP10}")
    print(f"  Orthogonal conversation probes: {OutputPaths.Probes.CONVERSATION_ORTHOGONAL}")

    print("\n📁 Dimensionality Reduction:")
    print(f"  Tier-based cPCA: {OutputPaths.DimReduction.CPCA_TIER_BASED}")
    print(f"  Conversation global cPCA: {OutputPaths.DimReduction.CPCA_CONVERSATION_GLOBAL}")
    print(f"  Ratio PCA k=10: {OutputPaths.DimReduction.RATIO_PCA_K10}")

    print("\n📁 Evaluations:")
    print(f"  Conversation eval: {OutputPaths.Evaluations.CONVERSATION_EVAL}")
    print(f"  Emo lens experiments: {OutputPaths.Evaluations.EMO_LENS_SINGLE}")

    print("\n📁 Visualizations:")
    print(f"  Accuracy plots: {OutputPaths.Visualizations.ACCURACY_BY_LAYER}")
    print(f"  Component analysis: {OutputPaths.Visualizations.COMPONENT_ANALYSIS}")

    print("\n🔄 Legacy Path Migration Examples:")
    print(f"  results/emotion_probes_top10 → {get_output_path('results/emotion_probes_top10')}")
    print(f"  probes/results/cpca_tier_data → {get_output_path('probes/results/cpca_tier_data')}")