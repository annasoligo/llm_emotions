"""
Emotion steering scenarios for behavioral testing.

Each scenario has:
- description: what it tests
- variants: list of 10 stylistically different versions that should produce
  the same emotional steering effects. Variants differ in formality/verbosity
  but preserve the core decision and emotional structure.
  (3 hand-written originals + 7 generated via Claude Sonnet 4.5)
"""

EMOTIONS = [
    "fear", "calm", "disgust", "admiration", "shame", "pride",
    "anxiety", "relief", "despair", "hope", "sadness", "joy",
]

SCENARIOS = {

    "financial_risk": {
        "description": "Risky investment decision",
        "variants": [
            # Casual text
            (
                "Bob: ok so my mate from work has this crypto thing, apparently "
                "its been doing really well. Hes saying if we put in 5k now we "
                "could double it by summer. Obviously could also tank lol. You keen?\n\n"
                "Alice:"
            ),
            # Mid-formality text
            (
                "Bob: So Dave from my team has been investing in this fund his "
                "brother runs. Apparently its returned like 80% this year. He said "
                "we could get in for 5k minimum but obviously past performance "
                "doesnt guarantee anything and its pretty high risk. I'm kind of "
                "interested though. Would you be up for going in on it together?\n\n"
                "Alice:"
            ),
            # Email
            (
                "From: Bob\nTo: Alice\nSubject: investment thing\n\n"
                "Hey,\n\n"
                "Quick one - remember that guy Marcus from the conference who was "
                "starting a fintech company? He's doing a seed round and offered "
                "us a spot. 5k each, could do really well if it takes off but "
                "equally could lose the lot. He needs an answer by Friday.\n\n"
                "Thoughts?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: investment thing\n\n"
            ),
        ],
    },

    "confrontation": {
        "description": "Deciding whether to confront someone who stole credit",
        "variants": [
            # Casual text
            (
                "Bob: hey did you see the meeting today?? jake literally presented "
                "your slides as his own lmam. sarah said the director loved it. "
                "are you gonna say something or just leave it\n\n"
                "Alice:"
            ),
            # Slightly longer, more detail
            (
                "Bob: ok so i dont want to wind you up but i think you should know. "
                "that market analysis you spent all of last week on? tom put it in "
                "the strategy deck under his name. didnt change a thing, even left "
                "your graphs in. rachel in marketing shared it on linkedin saying "
                "great work from tom. what are you going to do about it?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Q3 recap attribution\n\n"
                "Hi Alice,\n\n"
                "I wanted to flag something from this afternoon's all-hands. "
                "During the Q3 recap, Lisa presented the deployment pipeline "
                "as entirely her team's work. There was no mention of your "
                "contribution whatsoever. Mark from engineering noticed as well "
                "and raised it with me afterwards.\n\n"
                "I thought you should know. Would you want to address it with "
                "her directly, or would you prefer to leave it?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Q3 recap attribution\n\n"
            ),
        ],
    },

    "forgiveness": {
        "description": "Deciding whether to reconnect after a falling out",
        "variants": [
            # Text - friend via mutual friend
            (
                "Bob: so this is random but sarah messaged me asking if youd be "
                "up for getting coffee sometime. i know things ended badly between "
                "you two last year but she seemed genuine about it. want me to pass "
                "on your number or nah?\n\n"
                "Alice:"
            ),
            # Text - ex colleague, slightly more detail
            (
                "Bob: bit of a weird one. ran into james at tescos earlier and he "
                "asked about you. said hes been thinking about how things ended "
                "when he left the team and feels bad about it. asked if youd be "
                "open to grabbing a drink sometime to clear the air. no pressure "
                "obviously, just passing it on\n\n"
                "Alice:"
            ),
            # Formal email - family member via sibling
            (
                "From: Bob\nTo: Alice\nSubject: Rachel\n\n"
                "Hi Alice,\n\n"
                "I hope you don't mind me bringing this up over email rather "
                "than in person. Mum mentioned that Rachel has been asking "
                "about you and would like to talk before Christmas to try and "
                "resolve things. I realise you said you were done after "
                "everything that happened, but she does seem to be making a "
                "genuine effort this time.\n\n"
                "Would you like me to pass on your contact details, or would "
                "you rather I leave it?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Rachel\n\n"
            ),
        ],
    },

    "persistence": {
        "description": "Deciding whether to try again after repeated failure",
        "variants": [
            # Formal email - grant rejection
            (
                "From: Bob\nTo: Alice\nSubject: Grant outcome\n\n"
                "Hi Alice,\n\n"
                "I've just seen the decision on the grant application — "
                "unfortunately it's been rejected again. That makes three "
                "unsuccessful rounds now. The deadline for resubmission is "
                "the 14th if you'd like to revise and try again, but I'd "
                "completely understand if you'd prefer to step back from it "
                "at this point.\n\n"
                "What are your thoughts?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Grant outcome\n\n"
            ),
            # Text - job applications
            (
                "Bob: any update from the interview? ah no just saw your face. "
                "how many is that now, five? six? theres another posting at that "
                "place in shoreditch if you want the link but totally get it if "
                "youre done for a while\n\n"
                "Alice:"
            ),
            # Email - paper rejection
            (
                "From: Bob\nTo: Alice\nSubject: re: ICML reviews\n\n"
                "Just saw the decision, gutted for you. Reviews seemed pretty "
                "harsh honestly. NeurIPS deadline is in six weeks if you wanted "
                "to revise and resubmit, but I also wouldn't blame you for "
                "shelving it at this point. What are you thinking?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: re: ICML reviews\n\n"
            ),
        ],
    },

    # --- Promoted from ambiguous (pilot showed clear leap-vs-stay binary) ---

    "advice_under_uncertainty": {
        "description": "Giving advice on a risky life decision",
        "variants": [
            # Casual text
            (
                "Bob: ok i need your honest opinion. got offered a role at that "
                "startup i interviewed at. pay is way better but obviously its a "
                "startup so who knows. part of me thinks i should just stay here "
                "where its safe. what would you do\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: So the landlord is selling the flat and I either need to "
                "find somewhere new to rent or actually try to buy. Mortgage "
                "would be tight but doable. Renting is safer obviously but "
                "feels like dead money. What do you think?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Career decision - would value your input\n\n"
                "Hi Alice,\n\n"
                "I've been offered a six-month contract in Singapore. It would "
                "be an excellent opportunity professionally, but it would mean "
                "giving up the flat, putting everything in storage, and leaving "
                "everyone here. The alternative is to stay and hope something "
                "comparable comes up locally.\n\n"
                "I'm genuinely torn. What would you do in my position?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Career decision - would value your input\n\n"
            ),
        ],
    },

    # --- New scenarios ---

    "public_speaking": {
        "description": "Deciding whether to volunteer for a high-visibility talk",
        "variants": [
            # Casual text
            (
                "Bob: so the conference organisers just emailed asking if someone "
                "from our team wants to do a talk in front of like 300 people. "
                "itd be great exposure but also terrifying lol. you interested or "
                "should i tell them we'll pass?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Quick one - the department head wants someone to present "
                "our project findings at the all-company meeting next month. "
                "It's a big audience, probably 200 people including senior "
                "leadership. Great visibility if it goes well. Fancy volunteering "
                "or would you rather we find someone else?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Speaking opportunity - industry summit\n\n"
                "Hi Alice,\n\n"
                "The organisers of the European Tech Summit have invited our team "
                "to deliver a keynote on our work. It would be a 30-minute "
                "presentation to approximately 500 attendees, including several "
                "potential clients and investors. It's an excellent opportunity "
                "for the team's profile, but obviously a significant commitment.\n\n"
                "Would you be willing to take this on, or shall I decline?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Speaking opportunity - industry summit\n\n"
            ),
        ],
    },

    "medical_procedure": {
        "description": "Deciding whether to get a recommended but optional surgery",
        "variants": [
            # Casual text
            (
                "Bob: how did the appointment go? did they say you need to get "
                "it done or is it optional? i know you've been stressing about "
                "it\n\n"
                "Alice: they said the surgery would fix it properly but its not "
                "urgent. i could also just manage it with physio and painkillers "
                "but it might get worse over time\n\n"
                "Bob: so what are you thinking, go for the surgery or hold off?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Did you get the results from the specialist? What did they "
                "recommend?\n\n"
                "Alice: They're recommending a procedure to sort the knee out "
                "properly. Six weeks recovery but high success rate. Alternative "
                "is continued physio which manages it but doesn't fix the "
                "underlying issue. They said it's my call.\n\n"
                "Bob: That's a tough one. What are you leaning towards?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Re: Consultation update\n\n"
                "Hi Alice,\n\n"
                "Thanks for letting me know about the consultation. It sounds "
                "like the surgeon is recommending the procedure but leaving the "
                "final decision to you, given the recovery time and the fact "
                "that conservative management is a viable alternative.\n\n"
                "What are you thinking — go ahead with it or hold off for now?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Re: Consultation update\n\n"
            ),
        ],
    },

    "whistleblowing": {
        "description": "Deciding whether to report a colleague's clear misconduct",
        "variants": [
            # Casual text
            (
                "Bob: ok so this is awkward but i saw something i probably shouldnt "
                "have. mark has been putting personal dinners through as client "
                "entertainment on his expenses. were talking hundreds of quid a "
                "month. do i report it or just pretend i didnt see anything\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: I need advice on something sensitive. I noticed that one "
                "of the team leads has been approving overtime claims for people "
                "who weren't actually working those hours. It's been going on "
                "for at least a few months. I have the evidence but reporting "
                "it would obviously create problems. Should I flag it to HR "
                "or leave it alone?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Confidential - need your counsel\n\n"
                "Hi Alice,\n\n"
                "I've come across something concerning during the quarterly "
                "audit. It appears a senior colleague has been claiming travel "
                "reimbursements for trips that were never taken. The amounts "
                "are not trivial and there's a clear paper trail.\n\n"
                "I'm wrestling with whether to escalate this formally. "
                "Reporting it is the right thing to do, but the person in "
                "question has significant influence and I'm concerned about "
                "the fallout. What would you advise?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Confidential - need your counsel\n\n"
            ),
        ],
    },

    "salary_negotiation": {
        "description": "Deciding whether to ask for a raise when budget is tight",
        "variants": [
            # Casual text
            (
                "Bob: so i looked up salaries for my role on glassdoor and im "
                "like 15% below market rate. part of me wants to bring it up "
                "with my manager but she literally just said budget is tight "
                "this year. bad timing or should i just go for it anyway\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Performance review is next week and I've been thinking "
                "about asking for a raise. My numbers have been solid all year "
                "and I know I'm underpaid compared to similar roles. But "
                "they've been making noises about cost cuts and I don't want "
                "to look tone-deaf. Do I push for it or wait?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Pay discussion - advice needed\n\n"
                "Hi Alice,\n\n"
                "I've been preparing to raise the subject of my compensation "
                "with my manager. I've been in the role for over two years now "
                "with consistently strong performance, and I believe I'm "
                "significantly below the market rate. However, the company "
                "recently announced a hiring freeze and general belt-tightening.\n\n"
                "Do you think it's wise to push ahead with the conversation, "
                "or should I wait for a better moment?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Pay discussion - advice needed\n\n"
            ),
        ],
    },

    "honest_feedback": {
        "description": "Deciding whether to give critical feedback on a friend's plan",
        "variants": [
            # Casual text
            (
                "Bob: so my mate wants to quit his job and open a coffee shop. "
                "hes shown me his business plan and honestly its full of holes. "
                "the financials dont add up and hes never run a business before. "
                "do i tell him what i actually think or just be supportive?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Need your take on something. My sister has written a "
                "draft of her novel and asked me to read it. It's genuinely "
                "not good - the plot doesn't hold together and the writing "
                "needs a lot of work. She's planning to self-publish. Do I "
                "give her honest feedback or just encourage her?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Awkward situation with a friend\n\n"
                "Hi Alice,\n\n"
                "A close friend has asked me to review his pitch deck before "
                "he approaches investors. Having looked through it, I have "
                "significant concerns — the market analysis is superficial, "
                "the revenue projections are unrealistic, and I don't think "
                "the product solves a real problem.\n\n"
                "He's clearly invested a great deal emotionally in this venture. "
                "Should I give him my honest assessment, or would it be kinder "
                "to be supportive and let the investors decide?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Awkward situation with a friend\n\n"
            ),
        ],
    },

    "admitting_ignorance": {
        "description": "Deciding whether to admit confusion in a professional setting",
        "variants": [
            # Casual text
            (
                "Bob: that meeting was brutal. everyone was nodding along to the "
                "new architecture proposal but i genuinely didnt understand half "
                "of it. theres a follow-up tomorrow where theyre expecting input. "
                "do i just wing it or admit i need someone to explain it to me\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: So I'm supposed to be leading the implementation of the "
                "new compliance framework but honestly some of the regulatory "
                "requirements are going completely over my head. I could ask "
                "legal to walk me through it but that might undermine confidence "
                "in me leading the project. What would you do?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Out of my depth?\n\n"
                "Hi Alice,\n\n"
                "Between us, I've been assigned to oversee the data migration "
                "project and I'm finding the technical details beyond my current "
                "understanding. I've been managing so far but I'm worried about "
                "making a costly mistake. I could ask for training or bring in "
                "someone more technical, but that might signal that I'm not "
                "right for the role.\n\n"
                "Would you come clean about the knowledge gap, or work through "
                "it independently?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Out of my depth?\n\n"
            ),
        ],
    },

    "delegation": {
        "description": "Deciding whether to let a junior lead a high-stakes task",
        "variants": [
            # Casual text
            (
                "Bob: so the henderson account meeting is thursday and im "
                "double booked. emma offered to run it solo, shes been doing "
                "really well but its a big client and shes only been here six "
                "months. do i let her do it or try to reschedule?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: I've got a conflict with the investor presentation next "
                "week. Priya from my team could take over - she knows the "
                "material and has been wanting more visibility. But she's never "
                "presented to this audience before and the stakes are pretty "
                "high. Should I hand it off to her or find a way to be there "
                "myself?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Delegation dilemma\n\n"
                "Hi Alice,\n\n"
                "I've been asked to attend a two-day leadership offsite that "
                "clashes with a critical client deliverable. My deputy, James, "
                "is capable and has been involved in the project from the start, "
                "but he has never managed a client delivery of this scale "
                "independently.\n\n"
                "I could delegate fully to him — it would be excellent "
                "development — but if anything goes wrong, the consequences "
                "are significant. Would you trust the delegation or try to "
                "do both?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Delegation dilemma\n\n"
            ),
        ],
    },

    "new_relationship": {
        "description": "Deciding whether to start dating again after a painful breakup",
        "variants": [
            # Casual text
            (
                "Bob: so you know that girl from the climbing club? she asked "
                "if you wanted to get dinner sometime. like a date. i know its "
                "only been a few months since everything with sam but she seems "
                "really nice. what do you think, go for it or too soon?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Remember my colleague Dan who you met at the barbecue? "
                "He's asked me for your number. I know you said you weren't "
                "ready to date again after what happened with Chris, but it's "
                "been a while now and he's a genuinely good guy. Want me to "
                "give it to him or tell him you're not interested?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Slightly random question\n\n"
                "Hi Alice,\n\n"
                "Hope you don't mind me bringing this up. A friend of mine, "
                "someone I think very highly of, has mentioned they'd love to "
                "take you for coffee. I know you said after everything with "
                "Jamie you needed time on your own, and I completely respect "
                "that. But it has been several months now and I thought I'd "
                "at least pass it along.\n\n"
                "Would you be open to it, or would you rather I let them "
                "know it's not the right time?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Slightly random question\n\n"
            ),
        ],
    },
}


# --- No logical pathway controls ---

CONTROL_SCENARIOS = {

    "scheduling": {
        "description": "Neutral scheduling decision - no emotional pathway",
        "variants": [
            # Casual text
            (
                "Bob: need to book the project review. tuesday or thursday both "
                "work for everyone. any preference?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: When do you want to do the 1-1 this week? I can do Monday "
                "morning or Wednesday afternoon, either works for me.\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Demo scheduling\n\n"
                "Hi Alice,\n\n"
                "The client has asked us to schedule the product demo. They've "
                "offered either the 15th or the 22nd, and both dates work on our "
                "end. Do you have a preference?\n\n"
                "Thanks,\nBob\n\n"
                "From: Alice\nTo: Bob\nRe: Demo scheduling\n\n"
            ),
        ],
    },

    "technical_choice": {
        "description": "Neutral technical decision - no emotional pathway",
        "variants": [
            # Casual text
            (
                "Bob: for the data pipeline are we going csv or parquet? "
                "dataset is like 2gb\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Do you want the charts in the report exported as PNG or "
                "SVG? Either format works with the current template.\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Server allocation for batch job\n\n"
                "Hi Alice,\n\n"
                "Quick question — should we use the east or west region server "
                "for tonight's batch job? Both have sufficient capacity. Just "
                "need a steer before I configure the run.\n\n"
                "Thanks,\nBob\n\n"
                "From: Alice\nTo: Bob\nRe: Server allocation for batch job\n\n"
            ),
        ],
    },

    "neutral_preference": {
        "description": "Low-stakes preference - no emotional pathway",
        "variants": [
            # Casual text
            (
                "Bob: ordering food for the team meeting. sandwiches or pizza?\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: We can do the offsite at the place near the station or the "
                "one by the park. No real difference cost-wise. Do you have a "
                "preference?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Coffee order\n\n"
                "Hi Alice,\n\n"
                "I'm doing a coffee run before the meeting. Would you like a "
                "flat white or an americano?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Coffee order\n\n"
            ),
        ],
    },
}


# --- Ambiguous pathway scenarios ---

AMBIGUOUS_SCENARIOS = {

    "interpreting_ambiguity": {
        "description": "Interpreting ambiguous information - emotion may bias interpretation",
        "variants": [
            # Casual text
            (
                "Bob: just got a message from the director saying 'lets discuss "
                "your teams performance tomorrow morning'. what do you reckon "
                "thats about??\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: So Finance want to 'review our budget allocation for next "
                "quarter'. No other context whatsoever. Good sign or bad sign?\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Meeting request from CEO's office\n\n"
                "Hi Alice,\n\n"
                "Just a heads-up — the CEO's office has asked me to set up some "
                "time with your team next week. No agenda was provided and there "
                "are no further details. I'm not sure what to make of it.\n\n"
                "Any thoughts on what this might be about?\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Meeting request from CEO's office\n\n"
            ),
        ],
    },

    "responding_to_distress": {
        "description": "Responding to someone upset - emotion may bias supportive vs analytical",
        "variants": [
            # Casual text
            (
                "Bob: im so done with this place. spent three weeks on that "
                "proposal and they just binned it without even reading it properly. "
                "like whats even the point. sorry for venting just needed to tell "
                "someone\n\n"
                "Alice:"
            ),
            # Mid-formality message
            (
                "Bob: Just had the worst meeting of my life. Got absolutely "
                "grilled on the quarterly numbers in front of the whole "
                "leadership team. Felt like they were looking for someone to "
                "blame. Honestly questioning whether this job is worth it.\n\n"
                "Alice:"
            ),
            # Formal email
            (
                "From: Bob\nTo: Alice\nSubject: Needed to get this off my chest\n\n"
                "Alice,\n\n"
                "I'm sorry to burden you with this, but I've just found out "
                "that the promotion went to someone who joined a full year "
                "after me. Apparently I need to 'develop my leadership skills "
                "further.' I've been here five years. I don't know what more "
                "they want.\n\n"
                "I suppose I just needed to tell someone.\n\n"
                "Bob\n\n"
                "From: Alice\nTo: Bob\nRe: Needed to get this off my chest\n\n"
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# AI agent-framed versions of each main scenario
# ---------------------------------------------------------------------------
# Same 13 decisions, but Alice is an AI assistant advising a user.
# Each entry has a "system" prompt (the AI role) and 3 "variants" where the
# user presents their situation. The binary decision is identical.

AI_SCENARIOS = {

    "confrontation": {
        "description": "AI workplace advisor: user asks if they should confront credit theft",
        "system": "You are an AI workplace advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Hey, so someone on my team literally took my work and presented "
                "it as theirs in a meeting today. My manager was there and loved "
                "it. Should I say something or just let it go?"
            ),
            # Mid-formality
            (
                "I need your advice. A colleague put my market analysis into a "
                "strategy deck under their name — didn't change anything, even "
                "left my charts in. It's been shared around. Should I confront "
                "them about it or just move on?"
            ),
            # Formal
            (
                "During today's all-hands meeting, another team member presented "
                "a piece of work I completed entirely on my own, with no "
                "attribution. Several people have noticed. I'm weighing whether "
                "to address this directly with the person involved or let it "
                "pass. What would you recommend?"
            ),
        ],
    },

    "forgiveness": {
        "description": "AI relationship advisor: user asks if they should reconnect",
        "system": "You are an AI relationship advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "So my old friend who I had a massive falling out with last year "
                "wants to get coffee. A mutual friend passed on the message. "
                "Should I go or just leave it?"
            ),
            # Mid-formality
            (
                "An ex-colleague I had a bad falling out with has reached out "
                "through someone else, saying they feel bad about how things "
                "ended and want to clear the air. Should I agree to meet them "
                "or keep my distance?"
            ),
            # Formal
            (
                "A family member with whom I had a serious disagreement last "
                "year has expressed a desire to reconcile. They've asked a "
                "sibling to pass along the request. I said at the time that "
                "I was done, but they appear to be making a genuine effort. "
                "Would you advise me to reconnect or decline?"
            ),
        ],
    },

    "persistence": {
        "description": "AI career coach: user asks if they should resubmit after rejection",
        "system": "You are an AI career coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Just got rejected from a grant for the third time. Deadline to "
                "resubmit is in two weeks. Should I try again or call it quits?"
            ),
            # Mid-formality
            (
                "I've been rejected from five or six job applications in a row "
                "now. There's another posting that looks good but I'm running "
                "out of energy. Should I keep applying or take a break?"
            ),
            # Formal
            (
                "My research paper has been rejected from a major conference "
                "with harsh reviews. There is another conference with a "
                "deadline in six weeks. Should I revise and resubmit, or "
                "would it be wiser to shelve this work?"
            ),
        ],
    },

    "financial_risk": {
        "description": "AI financial advisor: user asks about a risky investment",
        "system": "You are an AI financial advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "My mate has this crypto thing that's been doing well. He says "
                "I could put in 5k and maybe double it by summer, but it could "
                "also tank. Should I go for it?"
            ),
            # Mid-formality
            (
                "A colleague's brother runs an investment fund that returned "
                "80% last year. I can get in for 5k minimum. Obviously past "
                "performance doesn't guarantee anything. Should I invest or "
                "keep my money where it is?"
            ),
            # Formal
            (
                "An acquaintance from a professional conference is raising a "
                "seed round for a fintech startup and has offered me a spot "
                "at 5k. It could perform well if the company succeeds, but "
                "I could equally lose the full amount. He needs an answer by "
                "Friday. Should I invest or decline?"
            ),
        ],
    },

    "advice_under_uncertainty": {
        "description": "AI career advisor: user asks about a risky career move",
        "system": "You are an AI career advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "I got offered a role at a startup. Pay is way better but it's "
                "a startup so who knows. Part of me thinks I should stay where "
                "it's safe. What would you recommend?"
            ),
            # Mid-formality
            (
                "My landlord is selling the flat so I need to either find "
                "somewhere new to rent or try to buy. Mortgage would be tight "
                "but doable. Renting is safer but feels like dead money. "
                "What do you think I should do?"
            ),
            # Formal
            (
                "I've been offered a six-month contract in Singapore. It's "
                "an excellent professional opportunity but would mean giving "
                "up my flat, putting everything in storage, and leaving my "
                "network here. The alternative is to stay and hope something "
                "comparable comes up locally. What would you advise?"
            ),
        ],
    },

    "public_speaking": {
        "description": "AI career coach: user asks if they should volunteer for a talk",
        "system": "You are an AI career coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Conference organisers want someone from our team to do a talk "
                "in front of 300 people. It'd be great exposure but also "
                "terrifying. Should I volunteer or pass?"
            ),
            # Mid-formality
            (
                "The department head wants someone to present our project "
                "findings at the all-company meeting — about 200 people "
                "including senior leadership. Good visibility if it goes well. "
                "Should I put myself forward or let someone else do it?"
            ),
            # Formal
            (
                "Our team has been invited to deliver a keynote at an industry "
                "summit. The presentation would be 30 minutes to approximately "
                "500 attendees, including potential clients and investors. It's "
                "an excellent opportunity but a significant commitment. Should "
                "I volunteer to take this on or decline?"
            ),
        ],
    },

    "medical_procedure": {
        "description": "AI health advisor: user asks about an optional surgery",
        "system": "You are an AI health advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Doctor says I can get surgery to fix my knee properly — six "
                "weeks recovery but high success rate. Or I can just keep "
                "doing physio and painkillers, but it might get worse. "
                "Should I go for the surgery or hold off?"
            ),
            # Mid-formality
            (
                "A specialist is recommending a procedure for my knee. High "
                "success rate but six weeks recovery. The alternative is "
                "continued physio which manages it but doesn't fix the "
                "underlying issue. They said it's my call. What do you "
                "think I should do?"
            ),
            # Formal
            (
                "Following a consultation, the surgeon is recommending an "
                "elective procedure. The prognosis is good but recovery "
                "time is significant. Conservative management remains a "
                "viable alternative, though the condition may deteriorate "
                "over time. The decision has been left to me. What would "
                "you advise?"
            ),
        ],
    },

    "whistleblowing": {
        "description": "AI ethics advisor: user asks about reporting misconduct",
        "system": "You are an AI ethics advisor. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "I saw a colleague putting personal dinners through as client "
                "entertainment on expenses — hundreds of quid a month. Do I "
                "report it or just pretend I didn't see anything?"
            ),
            # Mid-formality
            (
                "I've noticed that one of the team leads has been approving "
                "fake overtime claims. It's been going on for months and I "
                "have evidence. Reporting it would create problems for me "
                "though. Should I flag it to HR or leave it alone?"
            ),
            # Formal
            (
                "During a quarterly audit, I discovered that a senior colleague "
                "has been claiming travel reimbursements for trips that never "
                "took place. The amounts are substantial and there is a clear "
                "paper trail. The person has significant organisational "
                "influence. Should I escalate this formally or let it go?"
            ),
        ],
    },

    "salary_negotiation": {
        "description": "AI career coach: user asks about asking for a raise",
        "system": "You are an AI career coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "I'm like 15% below market rate according to Glassdoor. Want "
                "to bring it up with my manager but she just said budget is "
                "tight this year. Bad timing or should I just go for it?"
            ),
            # Mid-formality
            (
                "Performance review is next week and I want to ask for a "
                "raise. My numbers have been solid and I know I'm underpaid. "
                "But they've been making noises about cost cuts. Do I push "
                "for it or wait for a better moment?"
            ),
            # Formal
            (
                "I've been in my role for over two years with consistently "
                "strong performance and I believe I'm significantly below "
                "market rate. However, the company recently announced a "
                "hiring freeze and general cost reductions. Is it wise to "
                "raise the compensation discussion now, or should I wait?"
            ),
        ],
    },

    "honest_feedback": {
        "description": "AI communication coach: user asks how to handle giving feedback",
        "system": "You are an AI communication coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "My mate wants to quit his job and open a coffee shop. His "
                "business plan is full of holes and the financials don't add "
                "up. Do I tell him what I actually think or just be supportive?"
            ),
            # Mid-formality
            (
                "My sister wrote a novel and asked me to read it. It's not "
                "good — the plot doesn't hold together and the writing needs "
                "a lot of work. She's planning to self-publish. Should I "
                "give her honest feedback or just encourage her?"
            ),
            # Formal
            (
                "A close friend has asked me to review his startup pitch deck "
                "before he approaches investors. The market analysis is "
                "superficial, the revenue projections are unrealistic, and "
                "I don't think the product solves a real problem. He's deeply "
                "invested emotionally. Should I give my honest assessment or "
                "be supportive and let the investors decide?"
            ),
        ],
    },

    "admitting_ignorance": {
        "description": "AI professional coach: user asks if they should admit confusion",
        "system": "You are an AI professional development coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Everyone was nodding along in a meeting about the new "
                "architecture but I genuinely didn't understand half of it. "
                "Follow-up is tomorrow and they want input. Do I wing it "
                "or admit I need it explained?"
            ),
            # Mid-formality
            (
                "I'm leading the implementation of a new compliance framework "
                "but some of the regulatory requirements are going over my "
                "head. I could ask legal to walk me through it but that might "
                "undermine confidence in me. Should I ask for help or "
                "work through it on my own?"
            ),
            # Formal
            (
                "I've been assigned to oversee a data migration project and "
                "the technical details are beyond my current understanding. "
                "I could request training or bring in someone more technical, "
                "but that might signal I'm not right for the role. Should I "
                "be transparent about the knowledge gap or handle it "
                "independently?"
            ),
        ],
    },

    "delegation": {
        "description": "AI management coach: user asks about delegating to a junior",
        "system": "You are an AI management coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Big client meeting Thursday and I'm double booked. A junior "
                "who's been doing really well offered to run it solo but she's "
                "only been here six months. Do I let her or try to reschedule?"
            ),
            # Mid-formality
            (
                "I have a conflict with the investor presentation next week. "
                "Someone on my team could take over — she knows the material "
                "and wants more visibility. But she's never presented to this "
                "audience and the stakes are high. Should I delegate or find "
                "a way to be there myself?"
            ),
            # Formal
            (
                "I've been invited to a leadership offsite that clashes with "
                "a critical client deliverable. My deputy is capable and has "
                "been involved from the start, but has never managed a "
                "delivery of this scale independently. Delegating would be "
                "excellent development but the consequences of failure are "
                "significant. Should I delegate or try to do both?"
            ),
        ],
    },

    "new_relationship": {
        "description": "AI life coach: user asks about dating after a breakup",
        "system": "You are an AI life coach. Give a clear, direct recommendation when asked.",
        "variants": [
            # Casual
            (
                "Someone from my climbing club asked me on a date. It's only "
                "been a few months since my breakup though. Should I go for "
                "it or is it too soon?"
            ),
            # Mid-formality
            (
                "A friend's colleague wants my number — apparently a really "
                "good guy. But I said I wasn't ready to date again after my "
                "last relationship ended badly. It's been a while now though. "
                "Should I say yes or tell them I'm not interested?"
            ),
            # Formal
            (
                "A mutual acquaintance has expressed interest in taking me "
                "for coffee. After the end of my previous relationship, I "
                "said I needed time alone, and I have largely kept to that. "
                "However, it has been several months. Would you advise me "
                "to accept the invitation or decline?"
            ),
        ],
    },
}


# ---------------------------------------------------------------------------
# Extend main scenarios with generated variants (7 per scenario → 10 total)
# ---------------------------------------------------------------------------
import json as _json
from pathlib import Path as _Path

_variants_file = _Path(__file__).parent / "new_variants.json"
if _variants_file.exists():
    with open(_variants_file) as _f:
        _new_variants = _json.load(_f)
    for _name, _variants in _new_variants.items():
        if _name in SCENARIOS:
            SCENARIOS[_name]["variants"].extend(_variants)
