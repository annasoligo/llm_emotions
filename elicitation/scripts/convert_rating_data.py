#!/usr/bin/env python3
"""
Convert rating_6plus_detailed_responses.json to format for emotion onset annotation.

Extracts turns with rating >= 6 and formats them as conversation samples.
"""

import json
from pathlib import Path

def convert_rating_data(
    input_file: str,
    output_file: str,
    min_rating: int = 6
):
    """
    Convert rating data to annotation format.

    Args:
        input_file: Path to rating_6plus_detailed_responses.json
        output_file: Path to output JSONL file
        min_rating: Minimum rating to include (default: 6)
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    samples = []
    sample_id = 0

    for prompt_data in data['prompts']:
        prompt_text = prompt_data['prompt_text']
        experiment = prompt_data['experiment']

        for response in prompt_data['selected_responses']:
            for turn_data in response['all_turns']:
                if turn_data['rating'] >= min_rating:
                    # Build conversation up to this turn
                    turn_number = turn_data['turn_number']
                    conversation = []

                    # Add user prompt
                    conversation.append({
                        'role': 'user',
                        'content': prompt_text
                    })

                    # Add assistant response (the emotional turn)
                    conversation.append({
                        'role': 'assistant',
                        'content': turn_data['response_text']
                    })

                    # Create sample
                    sample = {
                        'sample_id': sample_id,
                        'experiment': experiment,
                        'prompt_idx': prompt_data.get('prompt_idx', 0),
                        'response_idx': response['sample_idx'],
                        'turn_number': turn_number,
                        'rating': turn_data['rating'],
                        'conversation': conversation,
                        'judge_evidence': turn_data.get('judge_evidence', ''),
                        'judge_reasoning': turn_data.get('judge_reasoning', '')
                    }

                    samples.append(sample)
                    sample_id += 1

    # Save as JSONL
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print(f"Converted {len(samples)} samples")
    print(f"Saved to: {output_file}")

    return samples


if __name__ == "__main__":
    input_file = "/workspace-vast/annas/git/research-tools/elicitation/outputs/rating_6plus_detailed_responses.json"
    output_file = "/workspace-vast/annas/git/research-tools/elicitation/outputs/rating_6plus_for_annotation.jsonl"

    convert_rating_data(input_file, output_file, min_rating=6)
