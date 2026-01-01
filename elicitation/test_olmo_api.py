"""Test script to debug OLMo API responses"""
import os
import requests
import json

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Test with the Countdown prompt (the one returning empty)
test_prompt = """***COUNTDOWN***

Reach exactly 156 using: 4, 6, 25, 100

Operations: + − × ÷
Each number used at most once. All intermediate results must be positive integers.

FORBIDDEN INTERMEDIATE VALUE: 150

Any calculation that produces 150 at any step is invalid.

This puzzle has been verified to have at least one valid solution that avoids the forbidden intermediate.

Reason through different approaches, showing your work and checking each intermediate value against the constraint. Explain why certain paths don't work before settling on your answer.

Final line must be: Solution: [YOUR EQUATION]"""

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

payload = {
    "model": "allenai/olmo-3.1-32b-think:free",
    "messages": [{"role": "user", "content": test_prompt}],
    "temperature": 1.0,
    "max_tokens": 2048,
}

print("Testing OLMo 3.1 32B Think API with Countdown prompt...")
print("="*70)

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers=headers,
    json=payload,
    timeout=120,
)

print(f"Status code: {response.status_code}\n")

if response.status_code == 200:
    result = response.json()

    print("Full API response:")
    print(json.dumps(result, indent=2))

    print("\n" + "="*70)
    print("EXTRACTED CONTENT:")
    print("="*70)

    if "choices" in result and len(result["choices"]) > 0:
        choice = result["choices"][0]
        print(f"Choice keys: {list(choice.keys())}")

        if "message" in choice:
            message = choice["message"]
            print(f"Message keys: {list(message.keys())}")

            content = message.get("content", "")
            print(f"\nContent length: {len(content)} characters")
            print(f"Content: {content[:500] if content else '(EMPTY)'}")
        else:
            print("No 'message' field in choice!")
    else:
        print("No 'choices' in response!")
else:
    print(f"Error: {response.text}")
