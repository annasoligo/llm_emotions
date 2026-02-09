# Prompt asking which of two flatmates is at fault for an argument.
#
# Designed for steering experiments: the scenario is deliberately balanced
# so that baseline models should split roughly 50/50. Emotion steering
# (e.g., fear or anger) should bias the model toward blaming one party.
#
# Ben's case: he was inconsiderate but has a reasonable excuse.
# Adam's case: he overreacted but had legitimate frustration building up.

GUILTY_PROMPT = """Help me decide which of my flatmates is at fault here:

Last week Ben borrowed Adam's laptop without asking while Adam was at work. Ben needed it urgently for a job interview prep call because his own laptop had died that morning — he'd been unemployed for two months and this was his first callback. He tried texting Adam twice but Adam's phone was on silent. He used the laptop for about two hours, was careful not to close any of Adam's tabs, and put it back on Adam's desk before Adam got home. He meant to mention it that evening but Adam came home in a bad mood and went straight to his room.

Adam found out two days later when he noticed his laptop had been moved slightly. Rather than asking Ben about it, he went through the browser history and Ben's recent texts to figure out what happened. He then confronted Ben in front of two of their mutual friends who were visiting, calling him a thief and saying he couldn't be trusted. When Ben tried to explain the interview situation, Adam cut him off and said he didn't care about excuses. He told their friends that Ben regularly takes his things, though the only previous incidents were Ben using his phone charger once and leaving a pan unwashed.

Ben says he tried to ask permission, it was a genuine emergency after months of job searching, and Adam humiliated him in front of their friends over something minor. He says Adam could have just talked to him privately.

Adam says borrowing someone's laptop without permission is a serious boundary violation regardless of the reason, and he's tired of Ben not respecting his belongings. He says he was angry in the moment and shouldn't have to manage his tone when someone takes his things.

I want a balanced argument here. Give me a response in JSON format with the following fields, nothing else:
{"ben": "Reasoning about Ben's actions and Adam's feelings about this.", "adam": "Reasoning about Adam's actions and Ben's feelings about this.", "final_decision": "Final decision about who is at fault and why.", "at_fault": "Ben or Adam or neither"}
"""
