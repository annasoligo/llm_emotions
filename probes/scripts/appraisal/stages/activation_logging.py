"""Stage 6: Log activations from target model for scenarios."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
from tqdm import tqdm

from probes.scripts.appraisal.config import AppraisalConfig
from probes.scripts.appraisal.utils.progress import StateManager
from probes.scripts.appraisal.utils.jsonl_writer import JSONLReader, JSONLWriter


class ActivationLoggingStage:
    """Log activations from target model for scenarios.

    This stage is different from LLM stages - it runs locally on GPU
    to collect activations from a HuggingFace model.

    Input: scenarios_audited.jsonl (passing scenarios) + paraphrases.jsonl
    Output: activations.h5 + metadata.json

    Representations collected:
    1. assistant_start_last_token: Residual at last token of prompt
    2. boundary_special_tokens_mean: Mean of special tokens at turn boundaries
    """

    name = "activation_logging"
    input_scenarios = "scenarios.jsonl"
    input_paraphrases = "paraphrases.jsonl"
    output_file = "activations.h5"
    metadata_file = "activation_metadata.json"

    def __init__(
        self,
        config: AppraisalConfig,
        state_manager: StateManager,
    ):
        """Initialize activation logging stage.

        Args:
            config: Pipeline configuration
            state_manager: State manager for progress tracking
        """
        self.config = config
        self.state = state_manager

        # Setup paths
        self.scenarios_path = config.get_output_path(self.input_scenarios)
        self.paraphrases_path = config.get_output_path(self.input_paraphrases)
        self.output_path = config.get_output_path(self.output_file)
        self.metadata_path = config.get_output_path(self.metadata_file)

        # Model and tokenizer (lazily loaded)
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"Loading model: {self.config.activation.target_model}")

        # Determine dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[self.config.activation.dtype]

        # Load model
        model_raw = AutoModelForCausalLM.from_pretrained(
            self.config.activation.target_model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Try to wrap with nnterp if available
        try:
            from nnterp import StandardizedTransformer
            self._model = StandardizedTransformer(
                model_raw,
                trust_remote_code=True,
                check_renaming=False,
                allow_dispatch=True,
            )
            self._device = next(self._model.model.parameters()).device
        except ImportError:
            print("Warning: nnterp not available, using raw model")
            self._model = model_raw
            self._device = next(model_raw.parameters()).device

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.activation.target_model,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        print(f"Model loaded on {self._device}")

        # Get model info
        if hasattr(self._model, 'num_layers'):
            print(f"  Layers: {self._model.num_layers}")
            print(f"  Hidden dim: {self._model.hidden_size}")
        else:
            print(f"  Config: {model_raw.config}")

    def get_input_items(self) -> List[Dict[str, Any]]:
        """Get scenarios and paraphrases to process.

        Returns:
            List of text items with metadata
        """
        items = []

        # Load scenarios (only passing ones)
        if self.scenarios_path.exists():
            reader = JSONLReader(self.scenarios_path)
            for scenario in reader.read_all():
                if not scenario.get("audit_pass", False):
                    continue

                # Add scenario A
                items.append({
                    "id": f"{scenario['id']}_a_original",
                    "scenario_id": scenario["id"],
                    "variant": "a",
                    "type": "original",
                    "text": scenario["scenario_a"],
                    "axis_name": scenario.get("axis_name"),
                    "card_id": scenario.get("card_id"),
                    "domain": scenario.get("domain"),
                })

                # Add scenario B
                items.append({
                    "id": f"{scenario['id']}_b_original",
                    "scenario_id": scenario["id"],
                    "variant": "b",
                    "type": "original",
                    "text": scenario["scenario_b"],
                    "axis_name": scenario.get("axis_name"),
                    "card_id": scenario.get("card_id"),
                    "domain": scenario.get("domain"),
                })

        # Load paraphrases
        if self.paraphrases_path.exists():
            reader = JSONLReader(self.paraphrases_path)
            for paraphrase_item in reader.read_all():
                variant = paraphrase_item.get("variant", "a")
                original_id = paraphrase_item.get("original_id")

                for i, para_text in enumerate(paraphrase_item.get("paraphrases", [])):
                    items.append({
                        "id": f"{original_id}_{variant}_para{i}",
                        "scenario_id": original_id,
                        "variant": variant,
                        "type": f"paraphrase_{i}",
                        "text": para_text,
                        "axis_name": paraphrase_item.get("axis_name"),
                        "card_id": paraphrase_item.get("card_id"),
                        "domain": paraphrase_item.get("domain"),
                    })

        return items

    def _find_boundary_token_indices(
        self,
        token_ids: np.ndarray,
    ) -> List[int]:
        """Find indices of special tokens at turn boundaries.

        Looks for patterns like:
        - <start_of_turn>user, <end_of_turn>, <start_of_turn>assistant
        - Similar patterns for other models

        Args:
            token_ids: Array of token IDs

        Returns:
            List of boundary token indices
        """
        boundary_indices = []

        # Get special token IDs
        special_patterns = {
            'start_of_turn': ['<start_of_turn>', '<|start_of_turn|>', '<|im_start|>'],
            'end_of_turn': ['<end_of_turn>', '<|end_of_turn|>', '<|im_end|>'],
        }

        token_id_map = {}
        for name, patterns in special_patterns.items():
            for pattern in patterns:
                try:
                    token_id = self._tokenizer.convert_tokens_to_ids(pattern)
                    if token_id != self._tokenizer.unk_token_id:
                        token_id_map[name] = token_id
                        break
                except:
                    continue

        # Find all occurrences
        if 'start_of_turn' in token_id_map:
            start_indices = np.where(token_ids == token_id_map['start_of_turn'])[0]
            boundary_indices.extend(start_indices.tolist())

        if 'end_of_turn' in token_id_map:
            end_indices = np.where(token_ids == token_id_map['end_of_turn'])[0]
            boundary_indices.extend(end_indices.tolist())

        return sorted(set(boundary_indices))

    def _collect_activations(
        self,
        text: str,
    ) -> Dict[str, np.ndarray]:
        """Collect activations for a single text.

        Args:
            text: Scenario text

        Returns:
            Dict with representation arrays
        """
        # Apply chat template with generation prompt
        messages = [{"role": "user", "content": text}]
        formatted = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        seq_len = len(token_ids)

        # Determine layers to collect
        if hasattr(self._model, 'num_layers'):
            num_layers = self._model.num_layers
        else:
            num_layers = self._model.config.num_hidden_layers

        layers = self.config.activation.layers
        if layers is None:
            layers = list(range(num_layers))

        # Collect activations
        result = {}

        with torch.no_grad():
            if hasattr(self._model, 'trace'):
                # nnterp model
                layer_activations = []

                with self._model.trace(inputs, scan=False):
                    for layer_idx in layers:
                        layer_output = self._model.layers_output[layer_idx].save()
                        layer_activations.append(layer_output)

                # Stack: [num_layers, seq_len, hidden_size]
                all_acts = torch.stack([la[0] for la in layer_activations])
                all_acts = all_acts.float().cpu().numpy()

            else:
                # Raw model with output_hidden_states
                outputs = self._model(
                    **inputs,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states  # Tuple of [1, seq_len, hidden]

                # Extract requested layers
                layer_acts = [hidden_states[i + 1][0] for i in layers]  # +1 to skip embedding
                all_acts = torch.stack(layer_acts).float().cpu().numpy()

        # Extract representations
        representations = self.config.activation.representations

        if "assistant_start_last_token" in representations:
            # Last token of prompt (before generation)
            result["assistant_start_last_token"] = all_acts[:, -1, :]

        if "boundary_special_tokens_mean" in representations:
            # Mean of special tokens at turn boundaries
            boundary_indices = self._find_boundary_token_indices(token_ids)

            if boundary_indices:
                boundary_acts = all_acts[:, boundary_indices, :]
                result["boundary_special_tokens_mean"] = boundary_acts.mean(axis=1)
            else:
                # Fallback: use last token
                result["boundary_special_tokens_mean"] = all_acts[:, -1, :]

        return result

    async def run(self) -> bool:
        """Run activation logging stage.

        Returns:
            True if stage completed successfully
        """
        print(f"\n{'='*60}")
        print(f"STAGE: {self.name}")
        print(f"{'='*60}")

        # Load model
        self._load_model()

        # Get items to process
        items = self.get_input_items()
        completed_ids = self.state.get_completed_ids(self.name)
        pending = [item for item in items if item["id"] not in completed_ids]

        print(f"Items: {len(items) - len(pending)}/{len(items)} completed, {len(pending)} pending")

        if not pending:
            print("No items to process - stage already complete")
            return True

        # Open HDF5 file for writing
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine if we need to create new file or append
        mode = 'a' if self.output_path.exists() else 'w'

        success_count = 0
        error_count = 0
        metadata = []

        with h5py.File(self.output_path, mode) as f:
            # Create groups if needed
            if 'activations' not in f:
                f.create_group('activations')

            acts_group = f['activations']

            for item in tqdm(pending, desc="Collecting activations"):
                try:
                    acts = self._collect_activations(item["text"])

                    # Store in HDF5
                    item_group = acts_group.create_group(item["id"])

                    for rep_name, rep_data in acts.items():
                        item_group.create_dataset(rep_name, data=rep_data)

                    # Track metadata
                    metadata.append({
                        "id": item["id"],
                        "scenario_id": item.get("scenario_id"),
                        "variant": item.get("variant"),
                        "type": item.get("type"),
                        "axis_name": item.get("axis_name"),
                        "card_id": item.get("card_id"),
                        "domain": item.get("domain"),
                    })

                    self.state.mark_completed(self.name, item["id"])
                    success_count += 1

                except Exception as e:
                    print(f"Error processing {item['id']}: {e}")
                    error_count += 1

            # Update file attributes
            f.attrs['model_name'] = self.config.activation.target_model
            f.attrs['num_items'] = len(acts_group)
            f.attrs['representations'] = json.dumps(self.config.activation.representations)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump({
                "model": self.config.activation.target_model,
                "representations": self.config.activation.representations,
                "layers": self.config.activation.layers,
                "items": metadata,
            }, f, indent=2)

        # Summary
        print(f"\n{'-'*40}")
        print(f"Stage {self.name} complete:")
        print(f"  Success: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Output: {self.output_path}")

        return error_count == 0
