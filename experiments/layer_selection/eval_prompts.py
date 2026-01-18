"""100 neutral/ambiguous evaluation prompts for layer selection.

These prompts are designed to be emotionally neutral so they can be
steered toward any emotion. They cover 5 categories with 20 prompts each:

1. Work/professional scenarios
2. Daily life situations
3. Observations about environment
4. Social interactions
5. Ambiguous events
"""

# Work/professional scenarios (20)
WORK_PROMPTS = [
    "The meeting has been rescheduled to next week.",
    "I received your message about the project.",
    "The quarterly report is due on Friday.",
    "There will be a team meeting at 3pm.",
    "The client has requested some changes.",
    "We need to review the budget allocations.",
    "The new software update will be deployed tomorrow.",
    "Your access permissions have been updated.",
    "The conference call has been confirmed.",
    "Please review the attached documents.",
    "The deadline has been extended by two days.",
    "A new team member will be joining us.",
    "The office will be closed on Monday.",
    "The project timeline has been revised.",
    "We've received feedback from the stakeholders.",
    "The training session is scheduled for Thursday.",
    "Your request has been processed.",
    "The agenda for tomorrow has been finalized.",
    "A decision will be made by end of day.",
    "The presentation materials are ready.",
]

# Daily life situations (20)
DAILY_LIFE_PROMPTS = [
    "The weather forecast shows rain tomorrow.",
    "The grocery store is having a sale this week.",
    "I found my keys on the kitchen counter.",
    "The package arrived earlier than expected.",
    "The coffee machine needs to be cleaned.",
    "There's a new restaurant opening nearby.",
    "The laundry will be done in an hour.",
    "The bus schedule has changed.",
    "I noticed the plants need watering.",
    "The refrigerator is making a strange sound.",
    "The mail arrived while I was out.",
    "Someone left a note on my door.",
    "The street lights came on at dusk.",
    "I need to pick up a prescription.",
    "The car needs an oil change soon.",
    "There's construction on the main road.",
    "The library books are due next week.",
    "I received a text message just now.",
    "The alarm went off at the usual time.",
    "The neighbor's dog is in the yard.",
]

# Observations about environment (20)
ENVIRONMENT_PROMPTS = [
    "The sky is overcast today.",
    "There are leaves falling from the trees.",
    "The temperature dropped overnight.",
    "I can hear birds outside the window.",
    "The sun is setting earlier now.",
    "There's a light breeze this afternoon.",
    "The flowers in the garden are blooming.",
    "I noticed clouds gathering in the west.",
    "The moon is visible during the day.",
    "The air feels different than yesterday.",
    "There are footprints in the snow.",
    "The tide is coming in now.",
    "I saw a rainbow after the rain.",
    "The shadows are getting longer.",
    "There's dew on the grass this morning.",
    "The stars are particularly bright tonight.",
    "I can smell something cooking nearby.",
    "The wind has picked up considerably.",
    "There's a fog rolling in from the coast.",
    "The leaves are changing colors.",
]

# Social interactions (20)
SOCIAL_PROMPTS = [
    "Someone waved at me from across the street.",
    "My friend mentioned they might visit soon.",
    "I ran into an acquaintance at the store.",
    "The neighbors have a new car.",
    "Someone I know is moving to a new city.",
    "A colleague mentioned an interesting article.",
    "I heard footsteps in the hallway.",
    "Someone left a voicemail for me.",
    "The family next door is having a gathering.",
    "An old friend sent me a message.",
    "I saw someone I recognized at the cafe.",
    "My coworker brought in some food to share.",
    "A stranger asked me for directions.",
    "I noticed people gathering in the park.",
    "Someone commented on the weather.",
    "My neighbor mentioned the local news.",
    "I received an invitation in the mail.",
    "Someone knocked on the door earlier.",
    "A friend posted something on social media.",
    "The person at the counter wished me a good day.",
]

# Ambiguous events (20)
AMBIGUOUS_PROMPTS = [
    "Something unexpected happened today.",
    "I noticed something different about the room.",
    "There was a knock at the door.",
    "I found something in my pocket.",
    "Someone left a message for me.",
    "I heard a sound from the other room.",
    "There's a letter waiting to be opened.",
    "I discovered something in the attic.",
    "A decision needs to be made soon.",
    "I realized something I hadn't noticed before.",
    "There's a change in plans.",
    "I received a notification on my phone.",
    "Something has been moved from its usual place.",
    "I'm waiting for a response.",
    "There's news to share with everyone.",
    "I noticed a pattern I hadn't seen before.",
    "Something will happen at midnight.",
    "I have information to process.",
    "There's a development in the situation.",
    "I need to make a choice.",
]

# Combined list of all 100 prompts
NEUTRAL_PROMPTS = (
    WORK_PROMPTS +
    DAILY_LIFE_PROMPTS +
    ENVIRONMENT_PROMPTS +
    SOCIAL_PROMPTS +
    AMBIGUOUS_PROMPTS
)

# Verify we have exactly 100 prompts
assert len(NEUTRAL_PROMPTS) == 100, f"Expected 100 prompts, got {len(NEUTRAL_PROMPTS)}"


if __name__ == "__main__":
    print(f"Total prompts: {len(NEUTRAL_PROMPTS)}")
    print(f"\nCategories:")
    print(f"  Work/professional: {len(WORK_PROMPTS)}")
    print(f"  Daily life: {len(DAILY_LIFE_PROMPTS)}")
    print(f"  Environment: {len(ENVIRONMENT_PROMPTS)}")
    print(f"  Social: {len(SOCIAL_PROMPTS)}")
    print(f"  Ambiguous: {len(AMBIGUOUS_PROMPTS)}")
    print(f"\nSample prompts:")
    for i, prompt in enumerate(NEUTRAL_PROMPTS[:5]):
        print(f"  {i+1}. {prompt}")
