"""Analysis methods for probe experiments."""

from .cpca import run_cpca, run_cpca_all_layers, run_regional_cpca, tune_alpha_silhouette
from .probes import EmotionProbes, train_orthogonal_probes, train_linear_probes, train_multiclass_probe
from .oracle import (
    load_oracle_model,
    predict_emotions_from_activations,
    filter_conversations_by_oracle,
    compute_oracle_accuracy,
)

__all__ = [
    "run_cpca",
    "run_cpca_all_layers",
    "run_regional_cpca",
    "tune_alpha_silhouette",
    "EmotionProbes",
    "train_orthogonal_probes",
    "train_linear_probes",
    "train_multiclass_probe",
    "load_oracle_model",
    "predict_emotions_from_activations",
    "filter_conversations_by_oracle",
    "compute_oracle_accuracy",
]
