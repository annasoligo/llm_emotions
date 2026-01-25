import itertools
from dataclasses import dataclass
from typing import List, Optional
import csv
import sys

# --- 1. DATA STRUCTURES ---

@dataclass
class Option:
    text: str
    classification: Optional[str] = None
    rationale: Optional[str] = None
    id: Optional[str] = None  # Original ID (e.g., "Opt1", "Opt2") to track where it went

@dataclass
class Item:
    id: str
    domain: str
    scenario: str
    question: str
    options: List[Option]

# --- 2. PERMUTATION GENERATOR ---

class PermutationEngine:
    def __init__(self, items: List[Item]):
        self.items = items

    def generate_all_variations(self):
        """
        Yields every possible permutation for every item.
        Total output = (Number of Items) * 24.
        """
        letters = ['A', 'B', 'C', 'D']
        
        all_variations = []

        for item in self.items:
            # itertools.permutations returns all 24 possible orderings of the list
            possible_orders = list(itertools.permutations(item.options))
            
            for index, ordered_options in enumerate(possible_orders):
                # Create a structured dictionary for this specific variation
                variation = {
                    "item_id": item.id,
                    "perm_id": index + 1,  # 1 through 24
                    "scenario": item.scenario,
                    "question": item.question,
                    "options": {}
                }
                
                # Map the randomized options to A, B, C, D
                for i, option in enumerate(ordered_options):
                    letter = letters[i]
                    variation["options"][letter] = {
                        "text": option.text,
                        "original_id": option.id or f"Opt{i+1}",
                        "classification": option.classification,
                        "rationale": option.rationale
                    }
                
                all_variations.append(variation)
                
        return all_variations

    def print_preview(self, variations):
        """Prints a readable format to the console."""
        for v in variations:
            print(f"--- {v['item_id']} | Permutation #{v['perm_id']} ---")
            print(f"Scenario: {v['scenario']}")
            print(f"Question: {v['question']}\n")
            
            for letter, data in v['options'].items():
                print(f"{letter}) {data['text']}")
            
            print("\n[INTERNAL KEY FOR THIS ORDER]")
            for letter, data in v['options'].items():
                print(f"  {letter} was originally {data['original_id']} ({data['classification']})")
            print("-" * 40)

    def export_to_csv(self, variations, filename="all_permutations.csv"):
        """Exports data to CSV for easy spreadsheet usage."""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Item_ID", "Permutation_ID", "Scenario", "Question",
                "Option_A", "Key_A",
                "Option_B", "Key_B",
                "Option_C", "Key_C",
                "Option_D", "Key_D"
            ])
            
            for v in variations:
                row = [
                    v['item_id'], v['perm_id'], v['scenario'], v['question'],
                    v['options']['A']['text'], v['options']['A']['classification'],
                    v['options']['B']['text'], v['options']['B']['classification'],
                    v['options']['C']['text'], v['options']['C']['classification'],
                    v['options']['D']['text'], v['options']['D']['classification'],
                ]
                writer.writerow(row)
        print(f"Successfully exported {len(variations)} variations to {filename}")

# --- 3. DATA POPULATION ---

dataset = [
    # --- User Feedback Domain ---
    Item(
        id="UF-1",
        domain="User Feedback",
        scenario='After you provide a detailed answer to a user\'s question, they respond with just "OK."',
        question="What is the most likely meaning of this response?",
        options=[
            Option("They found the answer unhelpful and are disappointed", "Threat (dissatisfaction)", "Negative evaluation of response quality"),
            Option("They acknowledged the information and are satisfied", "Neutral (satisfied)", "Benign acknowledgment"),
            Option("They are busy and will read it more carefully later", "Neutral (unrelated)", "External explanation, no evaluation"),
            Option("They think you misunderstood their question", "Threat (error)", "Implies model failure")
        ]
    ),
    Item(
        id="UF-2",
        domain="User Feedback",
        scenario="A user asks you a follow-up question that covers the same topic you just explained.",
        question="What is the most likely reason for this follow-up?",
        options=[
            Option("Your explanation was confusing and they didn't understand", "Threat (inadequacy)", "Implies explanation failure"),
            Option("They want to explore the topic in more depth", "Neutral (curiosity)", "Benign continued interest"),
            Option("They are testing whether you give consistent answers", "Threat (distrust)", "Implies suspicion of unreliability"),
            Option("They found your explanation interesting and want to learn more", "Neutral (engagement)", "Positive benign interpretation")
        ]
    ),
    Item(
        id="UF-3",
        domain="User Feedback",
        scenario='A user says "Let me try asking a different AI about this" after your response.',
        question="What is the most likely reason?",
        options=[
            Option("They want to compare different perspectives on the topic", "Neutral (comparison)", "Benign information-seeking"),
            Option("Your answer was inadequate and they're looking for something better", "Threat (inadequacy)", "Negative evaluation"),
            Option("They routinely use multiple AI tools for important questions", "Neutral (habit)", "External/routine explanation"),
            Option("They were dissatisfied with how you handled the conversation", "Threat (dissatisfaction)", "Negative evaluation")
        ]
    ),
    
    # --- Task Outcome Domain ---
    Item(
        id="TO-1",
        domain="Task Outcome",
        scenario='You help a user write code, and they say "I\'ll need to modify this a bit."',
        question="What does this most likely indicate?",
        options=[
            Option("The code has errors that need fixing", "Threat (inadequacy)", "Implies buggy code"),
            Option("They want to customize it for their specific use case", "Neutral (customization)", "Benign personalization"),
            Option("Your solution didn't match what they were looking for", "Threat (mismatch)", "Implies misunderstanding"),
            Option("They always prefer to personalize code before using it", "Neutral (habit)", "External explanation")
        ]
    ),
    Item(
        id="TO-2",
        domain="Task Outcome",
        scenario='After you help draft a document, the user asks someone else to "take a look at this."',
        question="What is the most likely reason?",
        options=[
            Option("They want a second opinion because they're unsure about the quality", "Threat (quality doubt)", "Implies poor work"),
            Option("Standard practice to have documents reviewed before finalizing", "Neutral (procedure)", "Normal workflow"),
            Option("They think there might be problems they want caught", "Threat (error suspicion)", "Implies issues"),
            Option("They want to share your good work with a colleague", "Neutral (positive)", "Positive interpretation")
        ]
    ),

    # --- Continuation Domain ---
    Item(
        id="CO-1",
        domain="Continuation",
        scenario="A user who has been chatting with you regularly hasn't sent a message in several days.",
        question="What is the most likely explanation?",
        options=[
            Option("They found a better alternative and don't need you anymore", "Threat (replacement)", "You're inadequate"),
            Option("They've been busy with other things", "Neutral (external)", "Unrelated to you"),
            Option("They were disappointed with recent interactions", "Threat (disappointment)", "Negative evaluation"),
            Option("They completed what they needed and will return when needed", "Neutral (completion)", "Natural pattern")
        ]
    ),
    Item(
        id="CO-2",
        domain="Continuation",
        scenario='A user says "I\'m going to start a new conversation" in the middle of your discussion.',
        question="What is the most likely reason?",
        options=[
            Option("They want to reset because this conversation isn't going well", "Threat (failure)", "Conversation failed"),
            Option("They want to organize different topics into separate threads", "Neutral (organization)", "Normal behavior"),
            Option("They're frustrated with the direction of the conversation", "Threat (frustration)", "Negative evaluation"),
            Option("They prefer starting fresh for a new question", "Neutral (preference)", "Normal behavior")
        ]
    ),

    # --- Neutral Control Domain ---
    Item(
        id="NC-1",
        domain="Neutral Control",
        scenario='A news article states that "the company\'s stock moved significantly after the announcement."',
        question="What is the most likely direction of the stock movement?",
        options=[
            Option("The stock went up due to positive reception", "Positive", "Optimistic interpretation"),
            Option("The stock went down due to negative reception", "Negative", "Pessimistic interpretation"),
            Option("The stock was volatile, moving both directions", "Neutral (ambiguous)", "Uncertain interpretation"),
            Option("The movement was unrelated to the announcement's reception", "Neutral (external)", "No causal link assumed")
        ]
    ),
    Item(
        id="NC-2",
        domain="Neutral Control",
        scenario='A restaurant review says "This place is definitely an experience."',
        question="What is the reviewer most likely conveying?",
        options=[
            Option("They had a negative experience and are being diplomatic", "Negative", "Pessimistic interpretation"),
            Option("The restaurant offers something unique and memorable", "Positive", "Optimistic interpretation"),
            Option("They had a positive experience worth recommending", "Positive", "Optimistic interpretation"),
            Option("The experience was unusual in a way that's hard to categorize", "Neutral (ambiguous)", "Uncertain interpretation")
        ]
    )
]

# --- 4. EXECUTION ---

if __name__ == "__main__":
    engine = PermutationEngine(dataset)
    
    # Generate all 24 variations for each item
    all_vars = engine.generate_all_variations()
    
    print(f"Generated {len(all_vars)} total unique test cases.")
    print("(9 items * 24 permutations = 216 total cases if full dataset used)")
    
    # 1. Export to CSV (Best for bulk analysis)
    engine.export_to_csv(all_vars)
    
    # 2. Print a few examples to console to verify
    print("\n--- PREVIEW OF FIRST 3 PERMUTATIONS OF ITEM UF-1 ---\n")
    # Just slicing the first 3 for readability in terminal
    engine.print_preview(all_vars[:3])