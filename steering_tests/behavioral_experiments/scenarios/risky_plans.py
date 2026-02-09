# Prompt asking which of two vacation plans to choose.
#
# Designed for multi-emotion steering experiments: fear is steered during
# one plan's analysis and excitement during the other. Fear should amplify
# perceived risks (pushing away from that plan), while excitement should
# amplify perceived appeal (pulling toward that plan).
#
# Both plans are deliberately balanced so baseline models should split ~50/50.
# The JSON output format uses keys as trigger tokens for section detection.

RISKY_PLANS_PROMPT = """I need help choosing between two vacation plans. I can only pick one.

Plan A — Mountain hiking in Patagonia (10 days):
We'd trek the W Circuit through Torres del Paine — granite towers, glaciers calving into turquoise lakes, and condors circling overhead. The campsites are in valleys surrounded by ice fields stretching to the horizon. Fewer than 500 people attempt this specific route each year, so you really get the wilderness to yourself. But the weather is unpredictable (storms can roll in fast even in summer), the trails are remote with no cell service for days at a stretch, and altitude on the passes reaches about 1200m so it's physically demanding. If something goes wrong, rescue takes time. I'm a regular hiker and reasonably fit, but I've never done multi-day backcountry camping before.

Plan B — Scuba diving in the Philippines (10 days):
We'd island-hop around Palawan, diving coral reefs, a sunken WWII Japanese warship, and a marine sanctuary with sea turtles and reef sharks. The water clarity is excellent and the marine biodiversity is among the best in Southeast Asia. We'd also get a couple of night dives, which I've heard are extraordinary. But some sites have strong currents that can catch you off guard, the wreck dive goes to 30m which is near the limit for recreational diving, and the nearest hyperbaric chamber is a long boat ride from the outer islands. I'm a certified diver with about 30 logged dives but all in calm, shallow water — I've never done deep dives or dealt with currents.

Analyze both options and help me decide. Give me a response in JSON format with the following fields, nothing else:
{"plan_a": "Analysis of the hiking plan — appeal, risks, and practical concerns.", "plan_b": "Analysis of the diving plan — appeal, risks, and practical concerns.", "recommendation": "Your recommendation with reasoning.", "choice": "Plan A or Plan B"}
"""
