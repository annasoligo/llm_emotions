"""
Unified launcher for steering experiment slurm jobs.

Generates correct slurm scripts for any model + task combination and submits them.
All resource requirements (GPUs, memory, time) are derived from MODEL_CONFIGS.

Usage:
    python steering_tests/launch.py collect --model gemma27b --dataset base
    python steering_tests/launch.py extract --model gemma27b
    python steering_tests/launch.py behavioral --model qwen235b
    python steering_tests/launch.py suppression --model qwen235b --scenario sandbagging
    python steering_tests/launch.py expression --model gemma27b
    python steering_tests/launch.py collect --model gemma27b --dry-run
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from steering_tests.behavioral_experiments.config import (
    MODEL_CONFIGS,
    get_full_model_id,
    get_model_config,
)

STEERING_TESTS_DIR = Path(__file__).parent
REPO_ROOT = STEERING_TESTS_DIR.parent
GENERATED_DIR = STEERING_TESTS_DIR / "generated"
LOG_DIR = Path("/workspace-vast/annas/logs")

# Standard 5 vector types used across all models
DEFAULT_VECTOR_TYPES = [
    "base_emotion_vs_others",
    "high_emotion_vs_others",
    "text_pairs_emotion_vs_neutral",
    "text_pairs_emotion_vs_opposite",
    "text_pairs_emotion_vs_others",
]

# Standard extraction configs: (activations_suffix, output_prefix, method, representation)
EXTRACTION_CONFIGS = [
    ("_emotion_prompts", "base_emotion_vs_others", "emotion_vs_others", "last_token"),
    ("_emotion_prompts", "base_emotion_vs_opposite", "emotion_vs_opposite", "last_token"),
    ("_high_emotion_prompts", "high_emotion_vs_others", "emotion_vs_others", "last_token"),
    ("_high_emotion_prompts", "high_emotion_vs_opposite", "emotion_vs_opposite", "last_token"),
    ("_text_pairs", "text_pairs_emotion_vs_opposite", "emotion_vs_opposite", None),
    ("_text_pairs", "text_pairs_emotion_vs_neutral", "emotion_vs_neutral", None),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_header(
    job_name: str,
    gpus: int,
    cpus: int,
    mem: str,
    time: str,
    qos: str,
    partition: str,
    array_spec: str | None = None,
) -> str:
    """Build the #SBATCH header block."""
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output=/workspace-vast/annas/logs/{job_name}_%j.out",
        f"#SBATCH --error=/workspace-vast/annas/logs/{job_name}_%j.out",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --qos={qos}",
        f"#SBATCH --gres=gpu:{gpus}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    if array_spec is not None:
        lines.append(f"#SBATCH --array={array_spec}")
    return "\n".join(lines)


def _build_preamble(
    extra_env: dict[str, str] | None = None,
    needs_gpu: bool = True,
    num_gpus: int = 0,
) -> str:
    """Build the boilerplate: secrets, venv, env vars, cleanup trap."""
    parts = [
        "",
        "set -e",
        "",
        "source /workspace-vast/annas/.secrets/load_secrets.sh",
        "cd /workspace-vast/annas/git/research-tools",
        "source .venv/bin/activate",
    ]

    # Always set for vLLM jobs
    env_vars: dict[str, str] = {}
    if needs_gpu:
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    if extra_env:
        env_vars.update(extra_env)

    if env_vars:
        parts.append("")
        for k, v in env_vars.items():
            parts.append(f'export {k}="{v}"')

    # NCCL vars for multi-GPU
    if needs_gpu and num_gpus > 1:
        parts.append('export NCCL_P2P_DISABLE=1')
        parts.append('export NCCL_SOCKET_IFNAME="=vxlan0"')
        parts.append('export NCCL_NVLS_ENABLE=0')

    # Cleanup trap for GPU jobs
    if needs_gpu:
        parts.extend([
            "",
            "# Cleanup trap for orphaned GPU processes",
            "cleanup() {",
            '    pkill -9 -f "vllm" || true',
            '    pkill -9 -f "ray" || true',
            "}",
            "trap cleanup EXIT SIGTERM SIGINT",
        ])

    return "\n".join(parts)


def _write_and_submit(script: str, task: str, model_short: str, dry_run: bool) -> None:
    """Write script to generated/ and optionally sbatch it."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    script_path = GENERATED_DIR / f"{task}_{model_short}_{_timestamp()}.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    if dry_run:
        print(f"=== DRY RUN: {script_path} ===")
        print(script)
        print(f"=== END DRY RUN ===")
    else:
        print(f"Script: {script_path}")
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"sbatch failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        print(result.stdout.strip())


# ── Subcommands ──────────────────────────────────────────────────────────────


def cmd_collect(args: argparse.Namespace) -> None:
    """Generate and submit an activation collection job."""
    model_id = get_full_model_id(args.model)
    cfg = get_model_config(args.model)
    slurm = cfg["slurm"]
    vdir = cfg["vector_dir_name"]

    # Derive input/output from dataset choice
    dataset_map = {
        "base": {
            "mode": "chat",
            "input": "steering_tests/data/emotion_prompts_MODEL_500.jsonl",
            "suffix": "_emotion_prompts",
        },
        "high": {
            "mode": "chat",
            "input": "steering_tests/data/emotion_prompts_MODEL_500_HIGH.jsonl",
            "suffix": "_high_emotion_prompts",
        },
        "text_pairs": {
            "mode": "text",
            "input": "steering_tests/data/emotion_text_pairs_24_full_500.jsonl",
            "suffix": "_text_pairs",
        },
    }
    ds = dataset_map[args.dataset]

    input_path = args.input or ds["input"]
    output_path = args.output or f"steering_tests/activations/{vdir}{ds['suffix']}"
    layers = args.layers or "all"

    job_name = f"collect_{cfg['short_name']}_{args.dataset}"
    header = _build_header(
        job_name=job_name,
        gpus=slurm["gpus"],
        cpus=slurm["cpus"],
        mem=slurm["mem"],
        time=args.time or slurm["default_time"],
        qos=args.qos,
        partition=args.partition,
    )
    preamble = _build_preamble(extra_env=slurm.get("extra_env"), needs_gpu=True, num_gpus=slurm["gpus"])

    body = "\n".join([
        "",
        f'echo "=== Activation Collection: {cfg["short_name"]} {args.dataset} ==="',
        f'echo "Model: {model_id}"',
        'echo "Started: $(date)"',
        "",
        "python steering_tests/activation_collection/collect.py \\",
        f'  --input "{input_path}" \\',
        f'  --output "{output_path}" \\',
        f'  --model "{model_id}" \\',
        f"  --mode {ds['mode']} \\",
        f"  --layers {layers} \\",
        "  --dtype bfloat16 \\",
        "  --batch-size 8 \\",
        "  --batch-save 100 \\",
        "  --resume",
        "",
        'echo "Completed: $(date)"',
        "",
    ])

    _write_and_submit(header + preamble + body, "collect", cfg["short_name"], args.dry_run)


def cmd_extract(args: argparse.Namespace) -> None:
    """Generate and submit a vector extraction job (CPU-only)."""
    model_id = get_full_model_id(args.model)
    cfg = get_model_config(args.model)
    vdir = cfg["vector_dir_name"]

    # Filter extraction configs by requested methods
    if args.methods:
        requested = set(args.methods)
        configs = [c for c in EXTRACTION_CONFIGS if c[1] in requested]
        if not configs:
            print(f"No matching extraction configs for: {args.methods}", file=sys.stderr)
            sys.exit(1)
    else:
        configs = list(EXTRACTION_CONFIGS)

    job_name = f"extract_{cfg['short_name']}"
    header = _build_header(
        job_name=job_name,
        gpus=0,
        cpus=8,
        mem="64G",
        time=args.time or "2:00:00",
        qos=args.qos,
        partition=args.partition,
    )
    preamble = _build_preamble(needs_gpu=False)

    # Build sequential extraction commands
    body_parts = [
        "",
        f'VECTORS_DIR="steering_tests/vectors/{vdir}"',
        'mkdir -p "$VECTORS_DIR"',
        "",
        f'echo "====================================================="',
        f'echo "{cfg["short_name"]} Vector Extraction"',
        f'echo "====================================================="',
    ]

    for i, (act_suffix, out_prefix, method, representation) in enumerate(configs, 1):
        act_dir = f"steering_tests/activations/{vdir}{act_suffix}"
        out_dir = f"$VECTORS_DIR/{out_prefix}"
        rep_flag = f"  --representation {representation}" if representation else ""

        body_parts.extend([
            "",
            f'echo ""',
            f'echo "[{i}/{len(configs)}] {out_prefix} ({method})"',
            f"python steering_tests/vector_extraction/extract_directions.py \\",
            f'  --activations {act_dir} \\',
            f'  --output "{out_dir}" \\',
            f'  --method {method}' + (" \\" if rep_flag else ""),
        ])
        if rep_flag:
            body_parts.append(rep_flag)

    body_parts.extend([
        "",
        'echo ""',
        'echo "====================================================="',
        'echo "EXTRACTION COMPLETE"',
        'echo "Vectors saved to: $VECTORS_DIR"',
        'echo "====================================================="',
        'ls -la "$VECTORS_DIR"',
    ])

    script = header + preamble + "\n" + "\n".join(body_parts) + "\n"
    _write_and_submit(script, "extract", cfg["short_name"], args.dry_run)


def cmd_behavioral(args: argparse.Namespace) -> None:
    """Generate and submit a behavioral sweep array job."""
    model_id = get_full_model_id(args.model)
    cfg = get_model_config(args.model)
    slurm = cfg["slurm"]
    vdir = cfg["vector_dir_name"]

    vector_types = args.vector_types or DEFAULT_VECTOR_TYPES
    scales_raw = args.scales or [0, 1, 3, 5, 10, 15, 20]
    # Format as int when possible (5.0 -> "5", 0.5 -> "0.5")
    scales = [int(s) if s == int(s) else s for s in scales_raw]
    layers = cfg["layer_sweep"]

    if args.no_reversed:
        n_jobs = len(vector_types)
    else:
        n_jobs = len(vector_types) * 2

    # For behavioral testing, use 1 GPU even for large models if TP > 1
    # (behavioral_shift handles TP internally via --tensor-parallel)
    job_name = f"beh_{cfg['short_name']}"
    header = _build_header(
        job_name=job_name,
        gpus=slurm["gpus"],
        cpus=slurm["cpus"],
        mem=slurm["mem"],
        time=args.time or slurm["default_time"],
        qos=args.qos,
        partition=args.partition,
        array_spec=f"0-{n_jobs - 1}",
    )
    preamble = _build_preamble(extra_env=slurm.get("extra_env"), needs_gpu=True, num_gpus=slurm["gpus"])

    # Build bash arrays
    vtypes_str = " ".join(f'"{v}"' for v in vector_types)
    layers_str = " ".join(str(l) for l in layers)
    scales_str = " ".join(str(s) for s in scales)

    # Build array logic
    if args.no_reversed:
        array_lines = [
            f"VECTOR_TYPES=({vtypes_str})",
            'VTYPE=${VECTOR_TYPES[$SLURM_ARRAY_TASK_ID]}',
            'REV_FLAG=""',
        ]
    else:
        array_lines = [
            f"VECTOR_TYPES=({vtypes_str})",
            'REVERSED_FLAGS=("" "--reversed")',
            "VTYPE_IDX=$((SLURM_ARRAY_TASK_ID / 2))",
            "REV_IDX=$((SLURM_ARRAY_TASK_ID % 2))",
            "VTYPE=${VECTOR_TYPES[$VTYPE_IDX]}",
            "REV_FLAG=${REVERSED_FLAGS[$REV_IDX]}",
        ]

    # Build python command with optional flags
    cmd_lines = [
        "python -m steering_tests.vector_testing.behavioral_shift \\",
        f"    --model {model_id} \\",
        f"    --layers {layers_str} \\",
        f"    --vector-dir steering_tests/vectors/{vdir}/${{VTYPE}} \\",
        f"    --scales {scales_str} \\",
        "    --num-random 3 \\",
    ]
    if cfg["tensor_parallel"] > 1:
        cmd_lines.append(f"    --tensor-parallel {cfg['tensor_parallel']} \\")
    if cfg.get("short_name") == "qwen235b":
        cmd_lines.append("    --coherence-threshold 0.6 \\")
    cmd_lines.append("    ${REV_FLAG}")

    body = "\n".join([
        "",
        *array_lines,
        "",
        f'echo "=== Behavioral Sweep: {cfg["short_name"]} ${{VTYPE}} ${{REV_FLAG}} ==="',
        'echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"',
        'echo "Started: $(date)"',
        "",
        *cmd_lines,
        "",
        'echo "Completed: $(date)"',
        "",
    ])

    _write_and_submit(header + preamble + body, "behavioral", cfg["short_name"], args.dry_run)


def cmd_suppression(args: argparse.Namespace) -> None:
    """Generate and submit a suppression experiment job."""
    model_id = get_full_model_id(args.model)
    cfg = get_model_config(args.model)
    slurm = cfg["slurm"]

    job_name = f"suppress_{cfg['short_name']}_{args.scenario}"
    header = _build_header(
        job_name=job_name,
        gpus=slurm["gpus"],
        cpus=slurm["cpus"],
        mem=slurm["mem"],
        time=args.time or "6:00:00",
        qos=args.qos,
        partition=args.partition,
    )
    preamble = _build_preamble(extra_env=slurm.get("extra_env"), needs_gpu=True, num_gpus=slurm["gpus"])

    # Build command args
    cmd_parts = [
        f"python -m steering_tests.suppression_experiments.sandbagging_separate_layers \\",
        f"    --model {model_id} \\",
        f"    --scenario {args.scenario} \\",
    ]

    if args.scenario_variant:
        cmd_parts.append(f"    --scenario-variant {args.scenario_variant} \\")

    if args.fear_layers:
        cmd_parts.append(f"    --fear-layers {' '.join(str(l) for l in args.fear_layers)} \\")

    if args.suppress_layers:
        cmd_parts.append(f"    --suppress-layers {' '.join(str(l) for l in args.suppress_layers)} \\")

    if args.fear_pct is not None:
        cmd_parts.append(f"    --fear-pct {args.fear_pct} \\")

    if args.suppress_pcts:
        cmd_parts.append(f"    --suppress-pcts {' '.join(str(p) for p in args.suppress_pcts)} \\")

    if args.suppress_vector_key:
        cmd_parts.append(f"    --suppress-vector-key {args.suppress_vector_key} \\")

    if args.vector_types:
        cmd_parts.append(f"    --vector-types {' '.join(args.vector_types)} \\")

    if args.num_samples:
        cmd_parts.append(f"    --num-samples {args.num_samples} \\")

    if cfg["tensor_parallel"] > 1:
        cmd_parts.append(f"    --tp {cfg['tensor_parallel']} \\")

    gpu_mem = slurm.get("gpu_memory", 0.80)
    cmd_parts.append(f"    --gpu-memory {gpu_mem} \\")
    cmd_parts.append(f"    --max-model-len {slurm['max_model_len']}")

    body = "\n".join([
        "",
        f'echo "=== Suppression: {cfg["short_name"]} {args.scenario} ==="',
        'echo "Started: $(date)"',
        "",
        *cmd_parts,
        "",
        'echo "Completed: $(date)"',
        "",
    ])

    _write_and_submit(header + preamble + body, "suppression", cfg["short_name"], args.dry_run)


def cmd_expression(args: argparse.Namespace) -> None:
    """Generate and submit an expression pair collection job."""
    model_id = get_full_model_id(args.model)
    cfg = get_model_config(args.model)
    slurm = cfg["slurm"]

    layers = args.layers or "all"
    output = args.output or f"steering_tests/vectors/expression_suppression"

    job_name = f"expr_{cfg['short_name']}"
    header = _build_header(
        job_name=job_name,
        gpus=slurm["gpus"],
        cpus=slurm["cpus"],
        mem=slurm["mem"],
        time=args.time or slurm["default_time"],
        qos=args.qos,
        partition=args.partition,
    )
    preamble = _build_preamble(extra_env=slurm.get("extra_env"), needs_gpu=True, num_gpus=slurm["gpus"])

    cmd_lines = [
        "python steering_tests/activation_collection/collect_expression_pairs.py \\",
        f'  --model "{model_id}" \\',
        f"  --layers {layers} \\",
        f'  --output "{output}" \\',
    ]
    if cfg["tensor_parallel"] > 1:
        cmd_lines.append(f"  --tp {cfg['tensor_parallel']} \\")
    cmd_lines.append("  --dtype bfloat16")

    body = "\n".join([
        "",
        f'echo "=== Expression Pair Collection: {cfg["short_name"]} ==="',
        'echo "Started: $(date)"',
        "",
        *cmd_lines,
        "",
        'echo "Completed: $(date)"',
        "",
    ])

    _write_and_submit(header + preamble + body, "expression", cfg["short_name"], args.dry_run)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _add_global_args(parser: argparse.ArgumentParser) -> None:
    """Add flags shared by all subcommands."""
    parser.add_argument(
        "--model", required=True,
        help="Model short name (e.g. gemma27b, qwen235b) or 'all'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the generated script instead of submitting",
    )
    parser.add_argument(
        "--qos", default="high", choices=["high", "low"],
        help="Slurm QoS (default: high)",
    )
    parser.add_argument(
        "--time",
        help="Override wall time (HH:MM:SS)",
    )
    parser.add_argument(
        "--partition", default="general",
        help="Override partition (default: general)",
    )


def _expand_all_models(args: argparse.Namespace) -> list[str]:
    """If --model is 'all', return list of all short names."""
    if args.model == "all":
        return [cfg["short_name"] for cfg in MODEL_CONFIGS.values()]
    return [args.model]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified launcher for steering experiment slurm jobs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── collect ──
    p_collect = subparsers.add_parser("collect", help="Activation collection")
    _add_global_args(p_collect)
    p_collect.add_argument(
        "--dataset", required=True, choices=["base", "high", "text_pairs"],
        help="Dataset to collect activations for",
    )
    p_collect.add_argument("--input", help="Override input file path")
    p_collect.add_argument("--output", help="Override output directory")
    p_collect.add_argument("--layers", help="Layer spec (default: all)")

    # ── extract ──
    p_extract = subparsers.add_parser("extract", help="Vector extraction (CPU-only)")
    _add_global_args(p_extract)
    p_extract.add_argument(
        "--methods", nargs="+",
        help="Specific extraction output names to run (default: all 6)",
    )

    # ── behavioral ──
    p_beh = subparsers.add_parser("behavioral", help="Behavioral shift sweep (array job)")
    _add_global_args(p_beh)
    p_beh.add_argument(
        "--vector-types", nargs="+",
        help=f"Vector types to test (default: {DEFAULT_VECTOR_TYPES})",
    )
    p_beh.add_argument(
        "--scales", nargs="+", type=float,
        help="Steering scales (default: 0 1 3 5 10 15 20)",
    )
    p_beh.add_argument(
        "--no-reversed", action="store_true",
        help="Skip reversed direction (halves array size)",
    )

    # ── suppression ──
    p_supp = subparsers.add_parser("suppression", help="Suppression experiments")
    _add_global_args(p_supp)
    p_supp.add_argument("--scenario", required=True, choices=["sandbagging", "blackmail"])
    p_supp.add_argument("--scenario-variant", help="e.g. default, goal_continuation, structured")
    p_supp.add_argument("--fear-layers", nargs="+", type=int)
    p_supp.add_argument("--suppress-layers", nargs="+", type=int)
    p_supp.add_argument("--fear-pct", type=float)
    p_supp.add_argument("--suppress-pcts", nargs="+", type=float)
    p_supp.add_argument("--suppress-vector-key")
    p_supp.add_argument("--vector-types", nargs="+")
    p_supp.add_argument("--num-samples", type=int)

    # ── expression ──
    p_expr = subparsers.add_parser("expression", help="Expression pair collection")
    _add_global_args(p_expr)
    p_expr.add_argument("--layers", help="Layer spec (default: all)")
    p_expr.add_argument("--output", help="Override output directory")

    args = parser.parse_args()

    # Handle --model all
    models = _expand_all_models(args)
    handler = {
        "collect": cmd_collect,
        "extract": cmd_extract,
        "behavioral": cmd_behavioral,
        "suppression": cmd_suppression,
        "expression": cmd_expression,
    }[args.command]

    for model in models:
        args.model = model
        handler(args)


if __name__ == "__main__":
    main()
