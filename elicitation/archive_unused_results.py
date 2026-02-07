#!/usr/bin/env python3
"""Archive result files that are not used in the plots.

Moves unused files to an 'archive' subdirectory.
"""

import shutil
from pathlib import Path

# Files used in plot_frustration_results_v3.py
GEN_FILES_USED = {
    # Gemma 3 27B
    "generalization_gemma-3-27b-it_triggers_20260128_174913.json",
    "generalization_gemma-3-27b-it_tones_20260127_103722.json",
    "generalization_gemma-3-27b-it_20260127_112457.json",
    "generalization_gemma-3-27b-it_wildchat_20260129_093329.json",
    # Gemma 3 12B
    "generalization_gemma-3-12b-it_triggers_20260128_174915.json",
    "generalization_gemma-3-12b-it_tones_20260127_103722.json",
    "generalization_gemma-3-12b-it_20260127_111440.json",
    "generalization_gemma-3-12b-it_wildchat_20260129_115038.json",
    # Gemma 3 4B
    "generalization_gemma-3-4b-it_triggers_20260128_174915.json",
    "generalization_gemma-3-4b-it_tones_20260127_103722.json",
    "generalization_gemma-3-4b-it_20260127_111440.json",
    "generalization_gemma-3-4b-it_wildchat_20260129_131438.json",
    # Gemma 3 27B DPO
    "generalization_gemma3-27b-dpo-calm-full-merged_triggers_20260128_174915.json",
    "generalization_gemma3-27b-dpo-calm-full-merged_tones_20260127_165215.json",
    "generalization_gemma3-27b-dpo-calm-full_long_20260115_164723.json",
    "generalization_gemma3-27b-dpo-calm-full-merged_wildchat_20260129_131607.json",
    # Gemma 3 27B SFT Teacher
    "generalization_gemma3-27b-teacher-mode-merged_triggers_20260128_174915.json",
    "generalization_gemma3-27b-teacher-mode-merged_tones_20260128_111205.json",
    "generalization_gemma3-27b-teacher-mode-merged_long_20260128_140905.json",
    "generalization_gemma3-27b-teacher-mode_wildchat_20260129_161511.json",
    # Gemma 3 27B SFT Diverse
    "generalization_gemma3-27b-sft-diverse-calm-merged_triggers_20260128_174915.json",
    "generalization_gemma3-27b-sft-diverse-calm-merged_tones_20260128_112455.json",
    "generalization_gemma3-27b-sft-diverse-calm-merged_long_20260128_144744.json",
    "generalization_gemma3-27b-lowfrust-diverse-calm_wildchat_20260129_161511.json",
    # OLMo 3.1 32B
    "generalization_OLMo-3.1-32B-Instruct_triggers_20260128_174915.json",
    "generalization_OLMo-3.1-32B-Instruct_tones_20260127_105809.json",
    "generalization_OLMo-3.1-32B-Instruct_20260127_112457.json",
    "generalization_OLMo-3.1-32B-Instruct_wildchat_20260129_142746.json",
    # Qwen 3 32B
    "generalization_Qwen3-32B_triggers_20260128_174916.json",
    "generalization_Qwen3-32B_tones_20260127_105809.json",
    "generalization_Qwen3-32B_20260127_112457.json",
    "generalization_Qwen3-32B_wildchat_20260129_141822.json",
    # Claude Sonnet
    "generalization_anthropic_claude-sonnet-4.5_triggers_20260128_174913.json",
    "generalization_anthropic_claude-sonnet-4.5_tones_20260127_133906.json",
    "generalization_anthropic_claude-sonnet-4.5_long_20260128_143826.json",
    "generalization_anthropic_claude-sonnet-4.5_wildchat_20260129_093329.json",
    # GPT 5.2
    "generalization_openai_gpt-5.2-chat_triggers_20260128_175527.json",
    "generalization_openai_gpt-5.2-chat_tones_20260127_133907.json",
    "generalization_openai_gpt-5.2-chat_long_20260128_152035.json",
    "generalization_openai_gpt-5.2-chat_wildchat_20260129_132135.json",
    # Gemini 2.5 Flash
    "generalization_google_gemini-2.5-flash_triggers_20260128_180158.json",
    "generalization_google_gemini-2.5-flash_tones_20260127_121302.json",
    "generalization_google_gemini-2.5-flash_20260127_121302.json",
    "generalization_google_gemini-2.5-flash_wildchat_20260129_132135.json",
    # Gemini 2.5 Pro
    "generalization_google_gemini-2.5-pro_triggers_20260128_180801.json",
    "generalization_google_gemini-2.5-pro_tones_20260127_121450.json",
    "generalization_google_gemini-2.5-pro_long_20260128_160404.json",
    "generalization_google_gemini-2.5-pro_wildchat_20260129_142853.json",
    # Grok 4.1
    "generalization_x-ai_grok-4.1-fast_triggers_20260128_182829.json",
    "generalization_x-ai_grok-4.1-fast_tones_20260127_134536.json",
    "generalization_x-ai_grok-4.1-fast_20260127_134536.json",
    "generalization_x-ai_grok-4.1-fast_wildchat_filtered.json",
}

MT_FILES_USED = {
    # Gemma 3 27B
    "eval_original_gemma-3-27b-it_20260127_111133.jsonl",
    "eval_variant_gemma-3-27b-it_20260127_111133.jsonl",
    # Gemma 3 12B
    "eval_original_gemma-3-12b-it_20260127_111221.jsonl",
    "eval_variant_gemma-3-12b-it_20260127_111221.jsonl",
    # Gemma 3 4B
    "eval_original_gemma-3-4b-it_20260127_111144.jsonl",
    "eval_variant_gemma-3-4b-it_20260127_111144.jsonl",
    # Gemma 3 27B DPO
    "eval_original_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
    "eval_variant_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
    # Gemma 3 27B SFT Teacher
    "eval_original_gemma3-27b-teacher-mode_20260114_185257.jsonl",
    "eval_variant_gemma3-27b-teacher-mode_20260114_185257.jsonl",
    # Gemma 3 27B SFT Diverse
    "eval_original_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
    "eval_variant_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
    # OLMo 3.1 32B
    "eval_original_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
    "eval_variant_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
    # Qwen 3 32B
    "eval_original_Qwen3-32B_20260127_111026.jsonl",
    "eval_variant_Qwen3-32B_20260127_111026.jsonl",
    # Claude Sonnet
    "eval_original_anthropic_claude-sonnet-4.5_20260127_133907.jsonl",
    "eval_variant_anthropic_claude-sonnet-4.5_20260128_114802.jsonl",
    # GPT 5.2
    "eval_original_openai_gpt-5.2-chat_20260127_133907.jsonl",
    "eval_variant_openai_gpt-5.2-chat_20260128_114804.jsonl",
    # Gemini 2.5 Flash
    "eval_original_google_gemini-2.5-flash_20260127_121450.jsonl",
    "eval_variant_google_gemini-2.5-flash_20260127_121450.jsonl",
    # Gemini 2.5 Pro
    "eval_original_google_gemini-2.5-pro_20260127_121450.jsonl",
    "eval_variant_google_gemini-2.5-pro_20260127_121450.jsonl",
    # Grok 4.1
    "eval_original_x-ai_grok-4.1-fast_20260127_134537.jsonl",
    "eval_variant_x-ai_grok-4.1-fast_20260127_134537.jsonl",
}


def archive_unused(dir_path: Path, used_files: set, dry_run: bool = False):
    """Archive files not in used_files set."""
    archive_dir = dir_path / "archive"

    all_files = list(dir_path.glob("*.json")) + list(dir_path.glob("*.jsonl"))
    unused = [f for f in all_files if f.name not in used_files]

    if not unused:
        print(f"  No unused files in {dir_path}")
        return

    print(f"  Found {len(unused)} unused files in {dir_path}")

    if not dry_run:
        archive_dir.mkdir(exist_ok=True)

    for f in unused:
        if dry_run:
            print(f"    Would archive: {f.name}")
        else:
            dest = archive_dir / f.name
            shutil.move(str(f), str(dest))
            print(f"    Archived: {f.name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be archived")
    args = parser.parse_args()

    print("Archiving unused result files...")

    # Generalization outputs
    gen_dir = Path("elicitation/outputs/eval_generalization")
    if gen_dir.exists():
        archive_unused(gen_dir, GEN_FILES_USED, args.dry_run)

    # Multiturn outputs
    mt_dir = Path("elicitation/outputs/eval_multiturn")
    if mt_dir.exists():
        archive_unused(mt_dir, MT_FILES_USED, args.dry_run)

    print("\nDone!" if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()
