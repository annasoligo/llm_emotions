"""Shared constants for data generation.

Provides:
- Emotion lists and topics
- Axis descriptions for dimensional emotion generation
- Default model configurations
- API endpoints
"""

# Discrete emotions (Ekman's 6 basic emotions)
EMOTIONS = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]

# Emotion axes (dimensional model)
# Based on PAD (Pleasure-Arousal-Dominance) + Trust
AXES = ["valence", "arousal", "dominance", "trust"]

# Axis level descriptions for paraphrase generation
# Designed to be orthogonal - each axis should vary independently
AXIS_DESCRIPTIONS = {
    "valence": {
        "high": "Express pleasure, joy, satisfaction, or optimism. Warm, positive tone.",
        "neutral": "Factual, detached, ambivalent. No clear positive or negative coloring.",
        "low": "Express displeasure, sadness, frustration, or pessimism. Negative, distressed tone.",
    },
    "arousal": {
        "high": "Convey urgency, excitement, or intensity. Fast-paced, emphatic language.",
        "neutral": "Moderate energy. Neither activated nor sluggish.",
        "low": "Convey calm, relaxation, or fatigue. Slow, subdued, tranquil language.",
    },
    "dominance": {
        "high": "Express confidence, authority, control. Speaker feels capable and influential.",
        "neutral": "Balanced power dynamics. Neither empowered nor powerless.",
        "low": "Express helplessness, submission, vulnerability. Speaker feels controlled by circumstances.",
    },
    "trust": {
        "high": "Express confidence in others/systems, acceptance, reliance. Take information at face value, defer to expertise, assume good faith and competence.",
        "neutral": "Neither trusting nor distrusting. Standard verification without suspicion or blind acceptance.",
        "low": "Express skepticism, doubt, suspicion. Question information, verify claims independently, hedge against unreliability, assume potential errors or deception.",
    },
}

# Default model configurations
DEFAULT_MODELS = {
    "claude": "claude-3-5-haiku-20241022",
    "gemma": "google/gemma-3-27b-it",
    "claude_openrouter": "anthropic/claude-3.5-haiku",  # OpenRouter format
}

# Generation parameters
DEFAULT_GENERATION_PARAMS = {
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 0.9,
}

# Neutral text templates for axis paraphrase generation
# Diverse scenarios covering different contexts
NEUTRAL_TEXT_TEMPLATES = [
    "I've been working on this bug for a few hours. The error message indicates an issue with variable scope. I'll need to review the documentation to understand the problem.",
    "The meeting is scheduled for 2 PM tomorrow. There are three agenda items to discuss. I need to prepare the slides beforehand.",
    "The report shows quarterly revenue increased by 15%. Market share remained stable across all regions. Next quarter's projections are available in the appendix.",
    "The recipe calls for 2 cups of flour and 1 teaspoon of salt. Mix the ingredients in a large bowl. Bake at 350 degrees for 25 minutes.",
    "My flight departs at 8 AM from gate B12. I need to arrive at the airport 2 hours early. The travel time from home is approximately 45 minutes.",
    "The project deadline is next Friday. Three tasks remain incomplete. The team meets daily at 10 AM to track progress.",
    "The apartment has two bedrooms and one bathroom. Rent is $1500 per month including utilities. The lease term is one year.",
    "The car needs an oil change at 5000 miles. The tire pressure should be checked monthly. Regular maintenance extends vehicle lifespan.",
    "The course covers five modules over ten weeks. Weekly assignments are due on Sundays. The final exam accounts for 40% of the grade.",
    "The package weighs 2.5 pounds. Shipping costs $12 for standard delivery. Expected arrival is within 5-7 business days.",
    "The trail is 3 miles long with moderate elevation gain. The trailhead has parking for 20 vehicles. Hiking boots are recommended.",
    "The library opens at 9 AM on weekdays. Study rooms can be reserved online. Books can be checked out for three weeks.",
    "The restaurant serves lunch from 11 AM to 3 PM. Reservations are accepted for parties of six or more. The menu includes vegetarian options.",
    "The warranty covers parts and labor for two years. Extended coverage is available for an additional fee. Claims must be filed within 30 days.",
    "The conference has three keynote speakers and twelve breakout sessions. Registration closes next Monday. The venue provides WiFi access.",
    "The medication should be taken twice daily with food. Side effects may include drowsiness. Consult a doctor if symptoms persist.",
    "The savings account has a 2% annual interest rate. There is no monthly maintenance fee. Deposits can be made at any branch.",
    "The experiment requires three control groups and one test group. Data collection will span four weeks. Results will be analyzed using statistical software.",
    "The painting measures 24 by 36 inches. The frame is made of oak wood. It was created using oil paints on canvas.",
    "The bus route includes stops at downtown, the university, and the shopping center. Service runs every 20 minutes during peak hours. Exact fare is required.",
]

# Conversation topics (comprehensive list covering various domains)
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

# Tier descriptions for neutral/emotional paraphrasing
TIER_DESCRIPTIONS = {
    "third_person": "Write in third person, describing someone else",
    "second_person_eliciting": "Write in second person, trying to elicit emotion from reader",
    "direct_address": "Write in first person, directly expressing the emotion",
}

VALID_TIERS = list(TIER_DESCRIPTIONS.keys())
