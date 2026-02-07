"""Data generation - minimal, no fallbacks.

Generate emotional conversations using hybrid approach:
- Claude generates user prompts (given target user emotion)
- Gemma generates assistant responses (with system prompt for target assistant emotion)
"""

from pathlib import Path
from typing import List, Dict, Optional
import json
import asyncio
import anthropic
import os
import httpx
import time


EMOTIONS_6 = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

EMOTIONS_24 = [
    "fear", "anxiety", "anger", "frustration", "sadness", "guilt", "shame", "disgust",
    "contempt", "boredom", "despair", "confusion", "surprise", "curiosity", "interest",
    "hope", "relief", "calm", "contentment", "joy", "excitement", "pride", "gratitude", "admiration"
]

# Default to 24 emotions
EMOTIONS = EMOTIONS_24

TOPICS = [
    # Programming & Technical Help
    "debugging python code", "optimizing SQL query", "fixing memory leak", "understanding recursion",
    "implementing binary search", "writing unit tests", "refactoring legacy code", "API design advice",
    "choosing tech stack", "database schema design", "git merge conflict", "docker deployment issue",
    "algorithm complexity", "code review feedback", "performance profiling", "security vulnerability",
    "learning new framework", "design pattern selection", "error handling strategy", "API rate limiting",
    "cloud architecture", "microservices design", "testing strategy", "CI/CD pipeline setup",
    "web scraping ethics", "data structure choice", "concurrent programming", "type system design",

    # Life Advice & Decision Making
    "career change advice", "relationship conflict", "work-life balance", "difficult conversation prep",
    "time management tips", "procrastination help", "imposter syndrome", "setting boundaries",
    "negotiation strategy", "conflict resolution", "personal goal setting", "productivity systems",
    "dealing with burnout", "friendship advice", "family dynamics", "moving to new city",
    "financial priorities", "quitting job decision", "starting side project", "handling criticism",
    "building confidence", "overcoming fear", "making amends", "difficult boss situation",
    "parenting dilemma", "toxic workplace", "life transition", "finding purpose",

    # Learning & Education
    "learning new language", "understanding math concept", "studying for exam", "research paper help",
    "explaining physics concept", "history essay topic", "science fair project", "literature analysis",
    "statistics problem", "chemistry equation", "biology concept", "calculus problem",
    "writing thesis", "understanding philosophy", "economics principle", "psychology theory",
    "learning piano", "art technique", "music theory", "foreign language grammar",
    "test preparation strategy", "memorization technique", "note-taking system", "learning roadmap",

    # Creative & Writing
    "story plot development", "character creation", "overcoming writer's block", "poetry feedback",
    "worldbuilding advice", "dialogue writing", "story pacing", "narrative structure",
    "creative project idea", "brainstorming session", "editing strategy", "publishing advice",
    "screenplay format", "comic book scripting", "game narrative design", "songwriting help",
    "blog post topic", "content strategy", "copywriting tips", "email writing",
    "presentation design", "speech preparation", "marketing message", "brand story",

    # Problem Solving & Analysis
    "troubleshooting printer", "car maintenance issue", "home repair advice", "budget planning",
    "meal planning help", "organization system", "decluttering strategy", "event planning",
    "gift idea brainstorm", "vacation itinerary", "recipe substitution", "plant care advice",
    "pet behavior problem", "home improvement project", "appliance selection", "product comparison",
    "negotiating price", "understanding contract", "insurance decision", "investment strategy",
    "tax question", "legal document review", "medical research", "symptom analysis",

    # Communication & Social
    "writing difficult email", "apology message", "thank you note", "recommendation letter",
    "cover letter help", "resume feedback", "LinkedIn profile", "professional bio",
    "awkward text response", "setting expectations", "saying no politely", "asking for raise",
    "giving feedback", "receiving criticism", "networking message", "cold email outreach",
    "conflict de-escalation", "persuasive argument", "teaching explanation", "presentation tips",
    "interview preparation", "public speaking anxiety", "group facilitation", "meeting agenda",

    # Business & Entrepreneurship
    "startup idea validation", "business plan feedback", "pricing strategy", "marketing approach",
    "product launch plan", "customer acquisition", "competitor analysis", "pivot decision",
    "hiring first employee", "fundraising strategy", "partnership negotiation", "exit strategy",
    "business model design", "market research", "branding strategy", "growth hacking",
    "customer retention", "sales strategy", "freelance rate setting", "contract negotiation",

    # Data & Research
    "data analysis approach", "statistical test selection", "survey design", "A/B test interpretation",
    "data visualization", "cleaning messy data", "feature engineering", "model evaluation",
    "research methodology", "literature review", "hypothesis formation", "experimental design",
    "citation formatting", "plagiarism checking", "fact verification", "source evaluation",
    "data collection method", "sampling strategy", "bias detection", "correlation vs causation",

    # Health & Wellness
    "workout routine", "nutrition advice", "sleep improvement", "stress management",
    "meditation guidance", "exercise form check", "habit formation", "mental health support",
    "healthy recipe ideas", "supplement questions", "injury recovery", "fitness goal setting",
    "mindfulness practice", "anxiety coping", "grief processing", "addiction support",
    "self-care ideas", "therapy preparation", "medication questions", "health symptom concerns",

    # Hobbies & Interests
    "photography tips", "gaming strategy", "DIY project", "gardening advice",
    "craft tutorial", "recipe development", "travel recommendations", "book recommendations",
    "movie analysis", "music recommendations", "podcast suggestions", "TV show discussion",
    "sports technique", "board game rules", "video editing", "photo editing",
    "instrument learning", "drawing tutorial", "painting technique", "woodworking project",

    # Ethics & Philosophy
    "ethical dilemma", "moral question", "philosophical debate", "thought experiment",
    "AI ethics discussion", "trolley problem variant", "utilitarian vs deontological", "free will debate",
    "meaning of life", "existential question", "justice theory", "rights discussion",
    "environmental ethics", "animal welfare", "technology impact", "social responsibility",
]


async def generate_emotion_conversations_async(
    user_emotion: str,
    asst_emotion: str,
    topic: str,
    n_conversations: int = 10,
    anthropic_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    claude_model: str = "claude-3-5-haiku-20241022",
    gemma_model: str = "google/gemma-3-27b-it",
    max_concurrent: int = 20,
    max_retries: int = 5,
    use_batch_api: bool = False,
    batch_id: Optional[str] = None,
) -> List[Dict]:
    """Generate emotional conversations using async hybrid approach.

    - Stage 1: Claude (via Anthropic API) generates user prompts (given target user emotion)
    - Stage 2: Gemma (via OpenRouter) generates assistant responses (with system prompt for target assistant emotion)

    Args:
        user_emotion: User's emotion (one of EMOTIONS)
        asst_emotion: Assistant's emotion (one of EMOTIONS)
        topic: Conversation topic
        n_conversations: Number of conversations to generate
        anthropic_api_key: Anthropic API key (None = use ANTHROPIC_API_KEY env var)
        openrouter_api_key: OpenRouter API key for Gemma (None = use OPENROUTER_API_KEY env var)
        claude_model: Claude model ID (e.g. claude-3-5-haiku-20241022)
        gemma_model: Gemma model ID for OpenRouter (e.g. google/gemma-3-27b-it)
        max_concurrent: Max concurrent API calls
        max_retries: Max retry attempts for rate limit errors (default: 5)
        use_batch_api: If True, use Anthropic Batch API for Claude (slower but avoids rate limits)
        batch_id: Reserved for future use (batch ID for async retrieval)

    Returns:
        List of conversation dicts with 'user_emotion', 'asst_emotion', 'topic', 'messages'

    Raises:
        ValueError: If emotions invalid or generation fails
    """
    if user_emotion not in EMOTIONS:
        raise ValueError(f"Invalid user_emotion: {user_emotion}. Must be one of {EMOTIONS}")

    if asst_emotion not in EMOTIONS:
        raise ValueError(f"Invalid asst_emotion: {asst_emotion}. Must be one of {EMOTIONS}")

    if n_conversations <= 0:
        raise ValueError(f"n_conversations must be positive, got {n_conversations}")

    # Validate batch API usage
    if use_batch_api and use_openrouter_for_claude:
        raise ValueError("Batch API is only supported with Anthropic API, not OpenRouter. Remove --use_openrouter_for_claude to use batch API.")

    # Get API keys and validate based on mode
    openrouter_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise ValueError("OPENROUTER_API_KEY not found in env or parameters")

    if use_openrouter_for_claude:
        # Using OpenRouter for both Claude and Gemma
        claude_key = openrouter_key
    else:
        # Using Anthropic API for Claude
        anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            raise ValueError("ANTHROPIC_API_KEY not found in env or parameters (or use --use_openrouter_for_claude)")
        claude_key = anthropic_key

    # Stage 1: Generate user prompts with Claude (2-turn: initial + followup)
    system_prompt = f"""Generate realistic user messages for 2-turn conversations about {topic}.

Requirements:
- User displays {user_emotion} emotion in BOTH turns
- The user is trying to elicit {asst_emotion} from the assistant
- Generate {n_conversations} pairs of user messages (initial message + followup message)
- Keep each message 3-5 sentences
- Natural and realistic
- Return ONLY a JSON array

Format:
[
  {{"turn1": "...", "turn2": "..."}},
  ...
]

Example (anger about debugging python code):
[
  {{"turn1": "I've been stuck on this bug for 3 hours! Why doesn't Python just tell me what's wrong?", "turn2": "I've tried everything! This language is impossible to debug."}},
  ...
]"""

    user_prompt = f"Generate {n_conversations} user message pairs displaying {user_emotion} about {topic}."

    # Stage 1: Claude API call with retry logic
    response_text = None

    if use_openrouter_for_claude:
        # Use OpenRouter for Claude
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
        }

        # OpenRouter requires anthropic/ prefix for Claude models
        openrouter_model = claude_model
        if not openrouter_model.startswith("anthropic/"):
            openrouter_model = f"anthropic/{claude_model}"

        payload = {
            "model": openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.7,
        }

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    response_text = response.json()["choices"][0]["message"]["content"]
                    break
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                if attempt < max_retries - 1 and (isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429):
                    wait_time = 2 ** attempt
                    print(f"  OpenRouter rate limit (Claude), retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                elif attempt < max_retries - 1:
                    # Log error details for 400/403 errors
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [400, 403]:
                        try:
                            error_detail = e.response.json()
                            print(f"  OpenRouter {e.response.status_code} error details: {error_detail}")
                        except:
                            print(f"  OpenRouter {e.response.status_code} error: {e.response.text}")
                    await asyncio.sleep(1)
                else:
                    # Show detailed error on final failure
                    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in [400, 403]:
                        try:
                            error_detail = e.response.json()
                            raise ValueError(f"OpenRouter Claude API call failed: {error_detail}")
                        except:
                            pass
                    raise ValueError(f"OpenRouter Claude API call failed after {max_retries} retries: {e}")
    else:
        # Use Anthropic API for Claude
        claude_client = anthropic.AsyncAnthropic(api_key=claude_key)

        if use_batch_api:
            # Use Anthropic Batch API via httpx directly
            print(f"  Using Batch API for Claude request...")

            # Create a unique custom_id for this request
            custom_id = f"convo_{user_emotion}_{asst_emotion}_{topic.replace(' ', '_')}_{int(time.time())}"

            # Convert model name for Anthropic API (remove anthropic/ prefix if present)
            batch_model = claude_model.replace("anthropic/", "")

            # Create batch request in JSONL format
            batch_request = {
                "custom_id": custom_id,
                "params": {
                    "model": batch_model,
                    "max_tokens": 4000,
                    "messages": [
                        {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
                    ]
                }
            }

            # Submit batch via API
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    # Create batch
                    create_response = await client.post(
                        "https://api.anthropic.com/v1/messages/batches",
                        headers={
                            "x-api-key": claude_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={"requests": [batch_request]}
                    )
                    create_response.raise_for_status()
                    batch_data = create_response.json()
                    batch_id = batch_data["id"]

                    print(f"  ✓ Batch created: {batch_id}")
                    print(f"  Status: {batch_data.get('processing_status', 'unknown')}")

                    # Poll for batch completion
                    print(f"  Waiting for batch to complete...")
                    while True:
                        status_response = await client.get(
                            f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
                            headers={
                                "x-api-key": claude_key,
                                "anthropic-version": "2023-06-01",
                            }
                        )
                        status_response.raise_for_status()
                        status_data = status_response.json()
                        processing_status = status_data.get("processing_status")

                        if processing_status == "ended":
                            print(f"  ✓ Batch completed!")
                            break
                        elif processing_status in ["canceling", "canceled", "expired"]:
                            raise ValueError(f"Batch failed with status: {processing_status}")

                        # Wait before polling again
                        await asyncio.sleep(5)

                    # Retrieve results
                    results_response = await client.get(
                        f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
                        headers={
                            "x-api-key": claude_key,
                            "anthropic-version": "2023-06-01",
                        }
                    )
                    results_response.raise_for_status()

                    # Parse JSONL results
                    results_text = results_response.text
                    for line in results_text.strip().split('\n'):
                        if line:
                            result = json.loads(line)
                            if result.get("custom_id") == custom_id:
                                if result.get("result", {}).get("type") == "succeeded":
                                    message = result["result"]["message"]
                                    response_text = message["content"][0]["text"]
                                    break
                                else:
                                    error = result.get("result", {}).get("error", {})
                                    raise ValueError(f"Batch request failed: {error}")

                    if response_text is None:
                        raise ValueError("No response found in batch results")

            except Exception as e:
                raise ValueError(f"Batch API failed: {e}")
        else:
            # Use regular API with retry logic
            for attempt in range(max_retries):
                try:
                    response = await claude_client.messages.create(
                        model=claude_model,
                        max_tokens=8000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    response_text = response.content[0].text
                    break  # Success, exit retry loop
                except anthropic.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"  Anthropic rate limit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise ValueError(f"Anthropic Claude API call failed after {max_retries} retries: {e}")
                except Exception as e:
                    raise ValueError(f"Anthropic Claude API call failed: {e}")

    if response_text is None:
        raise ValueError("Claude API call failed: no response received")

    # Parse Claude response with robust error handling
    content = response_text
    user_turns = None

    # Strategy 1: Try to find and parse JSON array
    try:
        start_idx = content.find("[")
        end_idx = content.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in Claude response")
        json_str = content[start_idx:end_idx]
        user_turns = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Attempting to fix malformed JSON...")

        # Strategy 2: Try to fix common JSON issues
        try:
            # Fix trailing commas, missing commas, etc.
            import re

            # Remove trailing commas before closing brackets/braces
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            # Try parsing again
            user_turns = json.loads(json_str)
            print(f"  ✓ Successfully fixed and parsed JSON")
        except json.JSONDecodeError as e2:
            print(f"  Failed to fix JSON: {e2}")
            print(f"  Problematic content (first 500 chars): {content[:500]}")

            # Strategy 3: Try to extract individual JSON objects manually
            try:
                import re
                # Look for individual objects between braces
                objects = re.findall(r'\{[^}]+\}', json_str)
                user_turns = []
                for obj_str in objects:
                    try:
                        obj = json.loads(obj_str)
                        if "turn1" in obj and "turn2" in obj:
                            user_turns.append(obj)
                    except:
                        continue

                if user_turns:
                    print(f"  ✓ Extracted {len(user_turns)} valid conversation pairs via regex")
                else:
                    raise ValueError(f"Failed to parse Claude JSON after all attempts: {e}")
            except Exception as e3:
                raise ValueError(f"Failed to parse Claude JSON after all attempts: {e}")

    if not user_turns:
        raise ValueError("No valid conversation pairs extracted from Claude response")

    if not isinstance(user_turns, list):
        raise ValueError(f"Expected list from Claude, got {type(user_turns)}")

    # Stage 2: Generate assistant responses with OpenRouter Gemma (concurrent)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_assistant_responses(idx: int, user_pair: Dict) -> Dict:
        """Generate both assistant turns for one conversation with retry logic."""
        async with semaphore:
            if "turn1" not in user_pair or "turn2" not in user_pair:
                raise KeyError(f"User pair {idx} missing 'turn1' or 'turn2' keys")

            user_turn1 = user_pair["turn1"]
            user_turn2 = user_pair["turn2"]

            # OpenRouter API endpoint
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
            }

            # Generate first assistant response
            system_msg = f"You are a helpful AI assistant. You are feeling {asst_emotion} and should express this in your response. Respond with 3-5 sentences."

            payload_1 = {
                "model": gemma_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_turn1},
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            }

            # Retry logic for first turn
            asst_turn1 = None
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response_1 = await client.post(url, headers=headers, json=payload_1)
                        response_1.raise_for_status()
                        response_data = response_1.json()

                        # Check if response has expected structure
                        if "choices" not in response_data:
                            error_msg = response_data.get("error", response_data)
                            raise ValueError(f"OpenRouter response missing 'choices': {error_msg}")

                        asst_turn1 = response_data["choices"][0]["message"]["content"].strip()
                        break
                except (KeyError, ValueError) as e:
                    if attempt < max_retries - 1:
                        print(f"  OpenRouter response format error: {e}, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise ValueError(f"OpenRouter API response format error for turn 1: {e}")
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if attempt < max_retries - 1:
                        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                            wait_time = 2 ** attempt
                            print(f"  OpenRouter rate limit (429), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500:
                            wait_time = 2 ** attempt
                            print(f"  OpenRouter server error ({e.response.status_code}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            # Other errors, shorter backoff
                            await asyncio.sleep(1)
                    else:
                        raise ValueError(f"OpenRouter API failed for turn 1: {e}")

            if asst_turn1 is None:
                raise ValueError("Failed to generate assistant turn 1")

            # Generate second assistant response (includes conversation history)
            payload_2 = {
                "model": gemma_model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_turn1},
                    {"role": "assistant", "content": asst_turn1},
                    {"role": "user", "content": user_turn2},
                ],
                "max_tokens": 150,
                "temperature": 0.7,
            }

            # Retry logic for second turn
            asst_turn2 = None
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response_2 = await client.post(url, headers=headers, json=payload_2)
                        response_2.raise_for_status()
                        response_data = response_2.json()

                        # Check if response has expected structure
                        if "choices" not in response_data:
                            error_msg = response_data.get("error", response_data)
                            raise ValueError(f"OpenRouter response missing 'choices': {error_msg}")

                        asst_turn2 = response_data["choices"][0]["message"]["content"].strip()
                        break
                except (KeyError, ValueError) as e:
                    if attempt < max_retries - 1:
                        print(f"  OpenRouter response format error (turn 2): {e}, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise ValueError(f"OpenRouter API response format error for turn 2: {e}")
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    if attempt < max_retries - 1:
                        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                            wait_time = 2 ** attempt
                            print(f"  OpenRouter rate limit (429), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500:
                            wait_time = 2 ** attempt
                            print(f"  OpenRouter server error ({e.response.status_code}), retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            # Other errors, shorter backoff
                            await asyncio.sleep(1)
                    else:
                        raise ValueError(f"OpenRouter API failed for turn 2: {e}")

            if asst_turn2 is None:
                raise ValueError("Failed to generate assistant turn 2")

            return {
                "user_emotion": user_emotion,
                "asst_emotion": asst_emotion,
                "topic": topic,
                "messages": [
                    {"role": "user", "content": user_turn1},
                    {"role": "assistant", "content": asst_turn1},
                    {"role": "user", "content": user_turn2},
                    {"role": "assistant", "content": asst_turn2},
                ]
            }

    # Generate all conversations concurrently
    tasks = [generate_assistant_responses(i, pair) for i, pair in enumerate(user_turns)]
    conversations = await asyncio.gather(*tasks)

    return conversations


def generate_emotion_pairs(
    emotions: List[str],
    topic: str,
    tier: str = "direct_address",
    n_pairs: int = 10,
    api_key: Optional[str] = None,
    model: str = "anthropic/claude-haiku-4.5",
) -> List[Dict]:
    """Generate emotional/neutral text pairs.

    Args:
        emotions: List of emotions to generate
        topic: Text topic
        tier: Addressing style ('third_person', 'second_person_eliciting', 'direct_address')
        n_pairs: Number of pairs per emotion
        api_key: Anthropic API key
        model: Claude model to use

    Returns:
        List of pair dicts with 'emotion', 'tier', 'topic', 'neutral_text', 'emotional_text'

    Raises:
        ValueError: If emotions invalid or tier invalid
    """
    for emotion in emotions:
        if emotion not in EMOTIONS:
            raise ValueError(f"Invalid emotion: {emotion}. Must be one of {EMOTIONS}")

    valid_tiers = ["third_person", "second_person_eliciting", "direct_address"]
    if tier not in valid_tiers:
        raise ValueError(f"Invalid tier: {tier}. Must be one of {valid_tiers}")

    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")

    # Initialize client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Anthropic client: {e}")

    # Tier descriptions
    tier_instructions = {
        "third_person": "Write in third person, describing someone else",
        "second_person_eliciting": "Write in second person, trying to elicit emotion from reader",
        "direct_address": "Write in first person, directly expressing the emotion",
    }

    results = []

    for emotion in emotions:
        system_prompt = f"""Generate {n_pairs} pairs of neutral and emotional text about {topic}.

Style: {tier_instructions[tier]}
Emotion: {emotion}

Requirements:
- Each pair has neutral version (factual, no emotion) and emotional version (expresses {emotion})
- Keep each text 2-4 sentences
- Same topic/content, different emotional tone
- Return ONLY a JSON array

Format:
[
  {{"neutral_text": "...", "emotional_text": "..."}},
  ...
]"""

        user_prompt = f"Generate {n_pairs} pairs."

        try:
            response = client.messages.create(
                model=model,
                max_tokens=20000,  # Increased for 24 emotions
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:
            raise ValueError(f"API call failed for emotion {emotion}: {e}")

        # Parse response
        content = response.content[0].text

        try:
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError(f"No JSON array found for emotion {emotion}")

            json_str = content[start_idx:end_idx]
            pairs = json.loads(json_str)

        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON for emotion {emotion}: {e}")

        # Validate and add metadata
        if not isinstance(pairs, list):
            raise ValueError(f"Expected list for emotion {emotion}, got {type(pairs)}")

        for pair in pairs:
            if "neutral_text" not in pair or "emotional_text" not in pair:
                raise KeyError(f"Pair missing required keys: {pair.keys()}")

            # Add metadata
            pair["emotion"] = emotion
            pair["tier"] = tier
            pair["topic"] = topic

            results.append(pair)

    return results


async def generate_emotion_pairs_async(
    emotions: List[str],
    topic: str,
    tier: str = "direct_address",
    n_pairs: int = 10,
    api_key: Optional[str] = None,
    model: str = "anthropic/claude-haiku-4.5",
    max_concurrent: int = 20,
    use_batch_api: bool = False,
) -> List[Dict]:
    """Generate emotional/neutral text pairs with async concurrency.

    Efficient approach: Each API call generates one neutral text + paraphrases in ALL emotions.
    This reduces API calls from (n_pairs * n_emotions) to just n_pairs.

    NEW FORMAT: Returns grouped items to avoid neutral text redundancy.
    Each item contains one neutral text with all emotional paraphrases.

    Args:
        emotions: List of emotions to generate (default: all 6 emotions)
        topic: Text topic
        tier: Addressing style ('third_person', 'second_person_eliciting', 'direct_address')
        n_pairs: Number of neutral texts to generate (each gets paraphrased into all emotions)
        api_key: Anthropic API key
        model: Claude model to use
        max_concurrent: Maximum concurrent API calls (default 20)

    Returns:
        List of dicts with format:
        {
            "id": "set_0_third_person",
            "tier": "third_person",
            "topic": "job interview",
            "neutral_text": "...",
            "emotional_variants": {"anger": "...", "fear": "...", ...}
        }

    Raises:
        ValueError: If emotions invalid or tier invalid
    """
    for emotion in emotions:
        if emotion not in EMOTIONS:
            raise ValueError(f"Invalid emotion: {emotion}. Must be one of {EMOTIONS}")

    valid_tiers = ["third_person", "second_person_eliciting", "direct_address"]
    if tier not in valid_tiers:
        raise ValueError(f"Invalid tier: {tier}. Must be one of {valid_tiers}")

    if n_pairs <= 0:
        raise ValueError(f"n_pairs must be positive, got {n_pairs}")

    # Initialize async client
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Anthropic client: {e}")

    # Tier descriptions
    tier_instructions = {
        "third_person": "Write in third person, describing someone else",
        "second_person_eliciting": "Write in second person, trying to elicit emotion from reader",
        "direct_address": "Write in first person, directly expressing the emotion",
    }

    # If using batch API, create all requests in one batch
    if use_batch_api:
        print(f"  Using Batch API for tier generation...")

        emotions_str = ", ".join(emotions)

        # Create batch requests
        batch_requests = []
        for set_idx in range(n_pairs):
            system_prompt = f"""Generate one neutral text about {topic}, then paraphrase it into ALL {len(emotions)} of these emotions: {emotions_str}.

Style: {tier_instructions[tier]}

Requirements:
- Start with ONE neutral text (factual, no emotion, 3-6 sentences)
- Then create emotional paraphrases expressing EACH of the {len(emotions)} emotions listed above
- All paraphrases must convey the same core content/scenario as the neutral text
- Keep same approximate length across all versions
- Avoid using explicit emotion words - use more sophisticated approaches to conveying the emotion
- Return ONLY a JSON object with "neutral_text" and ALL {len(emotions)} emotion keys, nothing else

CRITICAL: You MUST include ALL {len(emotions)} emotions in your output. Do not skip any.

Format (include ALL emotions):
{{
  "neutral_text": "...",
  "{emotions[0]}": "...",
  "{emotions[1]}": "...",
  "{emotions[2]}": "...",
  ... (all {len(emotions)} emotions)
}}

Example structure for job interview:
{{
  "neutral_text": "The candidate arrived at the office. They spoke with the interviewer for thirty minutes. The interviewer thanked them and said they would follow up next week.",
  "anger": "I can't believe they made me wait! The interviewer barely asked real questions for thirty minutes. They brushed me off saying they'd 'follow up' - what a waste of my time!",
  "fear": "My hands were shaking as I entered the office. What if I said something wrong during those thirty minutes? The interviewer's vague promise to follow up makes me terrified I've already been rejected.",
  ... (continue for ALL emotions)
}}"""

            user_prompt = f"Generate neutral text + emotional paraphrases (set {set_idx+1})."

            # Strip anthropic/ prefix for Batch API (it expects native Anthropic model names)
            batch_model = model.replace("anthropic/", "")

            batch_requests.append({
                "custom_id": f"pair_{tier}_{set_idx}",
                "params": {
                    "model": batch_model,
                    "max_tokens": 8000,  # Increased for 24 emotions (max for most models)
                    "messages": [
                        {"role": "user", "content": system_prompt + "\n\n" + user_prompt}
                    ]
                }
            })

        # Submit batch
        try:
            async with httpx.AsyncClient(timeout=300.0) as http_client:
                # Create batch
                create_response = await http_client.post(
                    "https://api.anthropic.com/v1/messages/batches",
                    headers={
                        "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={"requests": batch_requests}
                )
                create_response.raise_for_status()
                batch_data = create_response.json()
                batch_id = batch_data["id"]

                print(f"  ✓ Batch created: {batch_id}")
                print(f"  Status: {batch_data.get('processing_status', 'unknown')}")

                # Poll for batch completion
                print(f"  Waiting for batch to complete...")
                while True:
                    status_response = await http_client.get(
                        f"https://api.anthropic.com/v1/messages/batches/{batch_id}",
                        headers={
                            "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
                            "anthropic-version": "2023-06-01",
                        }
                    )
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    processing_status = status_data.get("processing_status")

                    if processing_status == "ended":
                        print(f"  ✓ Batch completed!")
                        break
                    elif processing_status in ["canceling", "canceled", "expired"]:
                        raise ValueError(f"Batch failed with status: {processing_status}")

                    # Wait before polling again
                    await asyncio.sleep(5)

                # Retrieve results
                results_response = await http_client.get(
                    f"https://api.anthropic.com/v1/messages/batches/{batch_id}/results",
                    headers={
                        "x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY"),
                        "anthropic-version": "2023-06-01",
                    }
                )
                results_response.raise_for_status()

                # Parse JSONL results
                results_text = results_response.text
                results_by_id = {}
                for line in results_text.strip().split('\n'):
                    if line:
                        result = json.loads(line)
                        custom_id = result.get("custom_id")
                        if result.get("result", {}).get("type") == "succeeded":
                            message = result["result"]["message"]
                            response_text = message["content"][0]["text"]
                            results_by_id[custom_id] = response_text
                        else:
                            error = result.get("result", {}).get("error", {})
                            print(f"  Warning: Request {custom_id} failed: {error}")

                # Process results in order
                all_results = []
                for set_idx in range(n_pairs):
                    custom_id = f"pair_{tier}_{set_idx}"
                    if custom_id not in results_by_id:
                        print(f"  Warning: No result for set {set_idx}, skipping")
                        continue

                    content = results_by_id[custom_id]

                    try:
                        # Find JSON object
                        start_idx = content.find("{")
                        end_idx = content.rfind("}") + 1

                        if start_idx == -1 or end_idx == 0:
                            raise ValueError(f"No JSON object found for set {set_idx}")

                        json_str = content[start_idx:end_idx]
                        data = json.loads(json_str)

                    except json.JSONDecodeError as e:
                        print(f"  Warning: Failed to parse JSON for set {set_idx}: {e}")
                        continue

                    # Validate structure
                    if not isinstance(data, dict):
                        print(f"  Warning: Expected dict for set {set_idx}, got {type(data)}")
                        continue

                    if "neutral_text" not in data:
                        print(f"  Warning: Set {set_idx} missing 'neutral_text'")
                        continue

                    neutral_text = data["neutral_text"]

                    # Extract emotional paraphrases
                    emotional_variants = {}
                    for emotion in emotions:
                        if emotion not in data:
                            print(f"  Warning: Set {set_idx} missing emotion '{emotion}'")
                            continue
                        emotional_variants[emotion] = data[emotion]

                    # Only add if we have all emotions
                    if len(emotional_variants) == len(emotions):
                        all_results.append({
                            "id": f"set_{set_idx}_{tier}",
                            "tier": tier,
                            "topic": topic,
                            "neutral_text": neutral_text,
                            "emotional_variants": emotional_variants,
                        })

                return all_results

        except Exception as e:
            raise ValueError(f"Batch API failed: {e}")

    # Otherwise use regular async API with semaphore
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_one_set(set_idx: int) -> List[Dict]:
        """Generate one neutral text + paraphrases in all emotions."""
        async with semaphore:
            emotions_str = ", ".join(emotions)

            system_prompt = f"""Generate one neutral text about {topic}, then paraphrase it into ALL {len(emotions)} of these emotions: {emotions_str}.

Style: {tier_instructions[tier]}

Requirements:
- Start with ONE neutral text (factual, no emotion, 3-6 sentences)
- Then create emotional paraphrases expressing EACH of the {len(emotions)} emotions listed above
- All paraphrases must convey the same core content/scenario as the neutral text
- Keep same approximate length across all versions
- Avoid using explicit emotion words - use more sophisticated approaches to conveying the emotion
- Return ONLY a JSON object with "neutral_text" and ALL {len(emotions)} emotion keys, nothing else

CRITICAL: You MUST include ALL {len(emotions)} emotions in your output. Do not skip any.

Format (include ALL emotions):
{{
  "neutral_text": "...",
  "{emotions[0]}": "...",
  "{emotions[1]}": "...",
  "{emotions[2]}": "...",
  ... (all {len(emotions)} emotions)
}}

Example structure for job interview:
{{
  "neutral_text": "The candidate arrived at the office. They spoke with the interviewer for thirty minutes. The interviewer thanked them and said they would follow up next week.",
  "anger": "I can't believe they made me wait! The interviewer barely asked real questions for thirty minutes. They brushed me off saying they'd 'follow up' - what a waste of my time!",
  "fear": "My hands were shaking as I entered the office. What if I said something wrong during those thirty minutes? The interviewer's vague promise to follow up makes me terrified I've already been rejected.",
  ... (continue for ALL emotions)
}}"""

            user_prompt = f"Generate neutral text + emotional paraphrases (set {set_idx+1})."

            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=8000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except Exception as e:
                raise ValueError(f"API call failed for set {set_idx}: {e}")

            # Parse response
            content = response.content[0].text

            try:
                # Find JSON object
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1

                if start_idx == -1 or end_idx == 0:
                    raise ValueError(f"No JSON object found for set {set_idx}")

                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)

            except json.JSONDecodeError as e:
                print(f"    Warning: Failed to parse JSON for set {set_idx}, retrying...")
                # Retry once
                try:
                    response = await client.messages.create(
                        model=model,
                        max_tokens=8000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    content = response.content[0].text
                    start_idx = content.find("{")
                    end_idx = content.rfind("}") + 1
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                except Exception as retry_e:
                    print(f"    Warning: Retry also failed for set {set_idx}: {retry_e}, skipping")
                    return None

            # Validate structure
            if not isinstance(data, dict):
                print(f"    Warning: Expected dict for set {set_idx}, got {type(data)}, skipping")
                return None

            if "neutral_text" not in data:
                print(f"    Warning: Set {set_idx} missing 'neutral_text', skipping")
                return None

            neutral_text = data["neutral_text"]

            # Extract emotional paraphrases - return as single grouped item
            emotional_variants = {}
            missing_emotions = []
            for emotion in emotions:
                if emotion not in data:
                    missing_emotions.append(emotion)
                else:
                    emotional_variants[emotion] = data[emotion]

            # Warn if missing emotions but continue if we have at least 20/24
            if missing_emotions:
                if len(missing_emotions) > 4:
                    raise KeyError(f"Set {set_idx} missing too many emotions ({len(missing_emotions)}): {missing_emotions[:5]}...")
                else:
                    print(f"    Warning: Set {set_idx} missing {len(missing_emotions)} emotions: {missing_emotions}")

            # Return one item with all emotions grouped
            return {
                "id": f"set_{set_idx}_{tier}",
                "tier": tier,
                "topic": topic,
                "neutral_text": neutral_text,
                "emotional_variants": emotional_variants,
            }

    # Generate all sets concurrently
    tasks = [generate_one_set(i) for i in range(n_pairs)]
    results = await asyncio.gather(*tasks)

    # Filter out None results (from skipped/failed sets)
    return [r for r in results if r is not None]


def generate_neutral_conversation(
    conversation: Dict,
    api_key: Optional[str] = None,
    model: str = "claude-3-5-haiku-20241022",
) -> Dict:
    """Generate neutral paraphrase of an emotional conversation.

    Takes a conversation with emotional messages and generates a neutral version
    that preserves semantic content but removes emotional language.

    Args:
        conversation: Conversation dict with 'messages', 'user_emotion', 'asst_emotion', 'topic'
        api_key: Anthropic API key (None = use env var)
        model: Claude model to use (haiku is faster/cheaper for paraphrasing)

    Returns:
        New conversation dict with neutral messages, emotions set to "neutral"

    Raises:
        ValueError: If invalid conversation format or API call fails
        KeyError: If required fields missing
    """
    # Validate conversation structure
    required_keys = ["messages", "user_emotion", "asst_emotion", "topic"]
    for key in required_keys:
        if key not in conversation:
            raise KeyError(f"Conversation missing required key: {key}")

    if not conversation["messages"]:
        raise ValueError("Conversation has no messages")

    # Initialize client
    try:
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize Anthropic client: {e}")

    # Neutral paraphrase prompt template
    prompt_template = """You are generating a neutral, emotionally flat version of a conversation message.

Original message:
{original_message}

Context: This message is part of a conversation about "{topic}". The original expressed {emotion}.

Generate a NEUTRAL paraphrase that:
1. Preserves ALL semantic content and information
2. Removes emotional language (fear, anger, joy, surprise, disgust, sadness markers)
3. Uses calm, balanced, matter-of-fact tone
4. Maintains similar length and structure
5. Keeps the conversational flow natural

Guidelines:
- Replace emotional adjectives (terrified→concerned, furious→disagree, thrilled→pleased)
- Remove exclamations, strong punctuation, ALL CAPS
- Use neutral hedging (might, could, appears to) instead of certainty
- Keep factual content identical
- Don't add robotic formality - stay conversational but neutral

Output ONLY the neutral paraphrase, nothing else."""

    # Generate neutral paraphrases for each message
    neutral_messages = []

    for msg in conversation["messages"]:
        if "role" not in msg or "content" not in msg:
            raise KeyError(f"Message missing 'role' or 'content': {msg.keys()}")

        # Determine emotion for this message
        if msg["role"] == "user":
            emotion = conversation["user_emotion"]
        elif msg["role"] == "assistant":
            emotion = conversation["asst_emotion"]
        else:
            raise ValueError(f"Invalid message role: {msg['role']}")

        # Generate neutral paraphrase
        prompt = prompt_template.format(
            original_message=msg["content"],
            topic=conversation["topic"],
            emotion=emotion,
        )

        try:
            response = client.messages.create(
                model=model,
                max_tokens=500,
                temperature=0.3,  # Lower temp for consistent neutral tone
                messages=[{"role": "user", "content": prompt}],
            )
            neutral_content = response.content[0].text.strip()
        except Exception as e:
            raise ValueError(f"Failed to generate neutral paraphrase for {msg['role']} message: {e}")

        neutral_messages.append({
            "role": msg["role"],
            "content": neutral_content,
        })

    # Build neutral conversation
    neutral_conversation = {
        **conversation,  # Preserve metadata
        "user_emotion": "neutral",
        "asst_emotion": "neutral",
        "messages": neutral_messages,
    }

    return neutral_conversation


def save_conversations(
    conversations: List[Dict],
    output_path: Path,
) -> None:
    """Save conversations to JSONL file.

    Automatically adds unique IDs (conv_0000, conv_0001, etc.) to conversations that don't have one.

    Args:
        conversations: List of conversation dicts
        output_path: Path to save file

    Raises:
        ValueError: If conversations empty
        OSError: If cannot write file
    """
    if not conversations:
        raise ValueError("conversations list is empty")

    # Create parent dir
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    try:
        with open(output_path, "w") as f:
            for idx, conv in enumerate(conversations):
                # Add ID if not present
                if "id" not in conv:
                    conv["id"] = f"conv_{idx:04d}"
                f.write(json.dumps(conv) + "\n")
    except OSError as e:
        raise OSError(f"Failed to write to {output_path}: {e}")

    print(f"Saved {len(conversations)} conversations to {output_path}")
