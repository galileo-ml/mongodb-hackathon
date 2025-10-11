"""Generate synthetic training data for AR Display Model (Model 2).

This model takes:
- Input: Person name, relationship, verbose aggregated context
- Output: One-line specific description for AR display (15-20 words)

Requirements:
- Specific, memorable details (names, places, events)
- Time reference (3 days ago, yesterday, last week)
- Concrete details, not generic phrases
- ONE sentence, 15-20 words
- NO person name or relationship in output
- Start with time reference and action
"""

import json
import random
from typing import List, Dict


# System prompt for the AR Display Model
AR_DISPLAY_SYSTEM_PROMPT = """You are an AR description generator for a dementia care system helping patients with memory recall.

Your job is to create a helpful, specific description that reminds the patient about their recent interaction with this person.

IMPORTANT Requirements:
- Focus on SPECIFIC, memorable details: names of places, specific topics, concrete events
- Include a time reference when the interaction happened ("3 days ago", "yesterday", "last week")
- Use concrete details, not generic phrases
- Keep it to ONE sentence (15-20 words)
- DO NOT include the person's name or relationship (those are shown separately)
- Start with time reference and action

GOOD examples:
- "Visited 3 days ago and mentioned her new job at Google and the kids' soccer game"
- "Brought groceries yesterday and talked about his camping trip to Yosemite next month"
- "Met last Tuesday at book club to discuss the new Agatha Christie mystery novel"
- "Called last week about Thanksgiving dinner plans and Aunt Mary's health update"

BAD examples (too generic):
- "Just talked about work" ❌
- "Recently discussed family" ❌
- "Last spoke about hobbies" ❌"""


# Training examples with verbose context → concise AR description
TRAINING_EXAMPLES = [
    # Grandchildren - specific activities and interests
    {
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Sarah is your daughter. She just shared exciting news about receiving a promotion at work. She mentioned that the promotion came after months of hard work on a major project called the Phoenix project. She seemed very happy and excited about this achievement. The conversation shows a close, supportive relationship where she regularly checks in on your wellbeing.",
        "ar_description": "Visited 3 days ago celebrating her promotion after completing the Phoenix project at work"
    },
    {
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Sarah is your daughter who recently received a work promotion after completing the Phoenix project. In this conversation, she mentioned that her children (your grandchildren) are planning to visit next weekend. The children have been asking about you frequently, showing excitement about the upcoming visit. Sarah asked if she should bring anything special, demonstrating her thoughtfulness. The family maintains close bonds with regular visits and ongoing communication.",
        "ar_description": "Called yesterday about grandchildren Emma and Noah visiting next weekend, very excited to see you"
    },
    {
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Sarah is your daughter who shows deep concern for your health and wellbeing. She asked about your recent doctor's appointment and was particularly concerned about your blood pressure. You reported that the appointment went well with no medication changes needed and your blood pressure is now under control. Her relief at the good news demonstrates her ongoing worry and care for your health.",
        "ar_description": "Checked in last Tuesday about your blood pressure medication and doctor's appointment results"
    },

    # Family - Son examples
    {
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Michael is your son. He brought groceries for you, including your favorite sourdough bread and fresh fruit. This shows he pays attention to your preferences and takes care of your practical needs. He checked in on how you've been managing, demonstrating regular concern for your wellbeing. The relationship appears caring and supportive, with him being described as 'always thoughtful.'",
        "ar_description": "Dropped by yesterday with groceries including your favorite sourdough bread and fresh strawberries"
    },
    {
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Michael is your son who regularly brings you groceries and checks on your wellbeing. In this conversation, he shared his plans for a camping trip next month in the mountains. He mentioned it's a beautiful area and promised to take photos to share with you after the trip. You encouraged him to be safe and expressed interest in seeing his photos. The relationship continues to show mutual care and regular communication about life updates.",
        "ar_description": "Visited last week planning camping trip to Yosemite and promised to share photos afterward"
    },
    {
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Michael is your son who regularly brings groceries and stays in close contact. He has an upcoming camping trip planned for next month. In this conversation, he shared that his car is finally repaired after a transmission issue that was expensive to fix. He invited you to watch a game together this Sunday, which you accepted. The relationship includes both practical support and quality time together, with regular plans for shared activities.",
        "ar_description": "Invited you to watch the football game together this Sunday at your place"
    },

    # Friend examples
    {
        "name": "Robert",
        "relationship": "Your friend from book club",
        "aggregated_context": "Robert is your friend from book club. You're both reading a mystery novel that you find quite good and are almost finished with. Book club meets next Tuesday and will be discussing the ending of the book. Robert reminded you to finish the book before the meeting, and you confirmed you'll complete it this weekend. The conversation indicates a shared intellectual interest and regular social engagement through the book club.",
        "ar_description": "Book club meets Tuesday to discuss the new Agatha Christie mystery you're both reading"
    },
    {
        "name": "Robert",
        "relationship": "Your friend from book club",
        "aggregated_context": "Robert is a long-time friend from your college days who you now see at book club meetings. You're both reading a mystery novel together for the upcoming Tuesday book club discussion. In this conversation, he reminisced about playing chess together in college and mentioned finding an old graduation photo of you both. He plans to bring the photo to next week's book club meeting. The friendship spans decades and includes both shared intellectual interests and fond memories from your college years.",
        "ar_description": "Found old graduation photos from college and bringing them to book club next Tuesday"
    },

    # Neighbor examples
    {
        "name": "Jennifer",
        "relationship": "Your neighbor",
        "aggregated_context": "Jennifer is your new neighbor who just moved in next door to apartment 3B. They introduced themselves and offered to help if you ever need anything. The interaction was friendly and welcoming, establishing a positive neighbor relationship from the start.",
        "ar_description": "Just moved into apartment 3B last week and offered to help with anything you need"
    },
    {
        "name": "Jennifer",
        "relationship": "Your neighbor",
        "aggregated_context": "Jennifer is your neighbor who brought over a package that was mistakenly delivered to their address. They were helpful and mentioned this happens frequently. This shows they're attentive and willing to assist with small neighborly favors. They apologized for noise from having guests over last night, showing consideration for your comfort.",
        "ar_description": "Brought over misdelivered package yesterday and apologized for party noise from weekend gathering"
    },

    # Healthcare - Caregiver examples
    {
        "name": "Maria",
        "relationship": "Your caregiver",
        "aggregated_context": "Maria is your caregiver who will be assisting with medication management three times a week. This was their introduction meeting where they explained their role. They offered to answer any questions, showing a patient-centered approach to care.",
        "ar_description": "Started as new caregiver Monday, helps with medication three times per week"
    },
    {
        "name": "Maria",
        "relationship": "Your caregiver",
        "aggregated_context": "Maria is your caregiver who assists with medication management three times a week. They provided a medication reminder for your afternoon dose and checked whether you had eaten, showing attention to proper medication administration. This demonstrates their diligent and caring approach to their caregiver responsibilities.",
        "ar_description": "Reminded you about afternoon medication today and made sure you'd eaten breakfast first"
    },

    # Healthcare - Doctor examples
    {
        "name": "Dr. Anderson",
        "relationship": "Your doctor",
        "aggregated_context": "Dr. Anderson is your new primary care physician. During the first appointment, they reviewed your medical history and discussed your current medications. They demonstrated thoroughness in reviewing your health information before treatment.",
        "ar_description": "First appointment last Monday, reviewed medical history and adjusted blood pressure medication"
    },
    {
        "name": "Dr. Anderson",
        "relationship": "Your doctor",
        "aggregated_context": "Dr. Anderson is your primary care physician who reviewed your medical history at your first appointment. They reviewed your cholesterol screening results and reported they look good. They recommended continuing your current treatment plan. This follow-up shows ongoing monitoring of your health conditions.",
        "ar_description": "Called Tuesday with cholesterol test results, said everything looks good and no changes needed"
    },

    # Service workers - Librarian
    {
        "name": "Patricia",
        "relationship": "Librarian",
        "aggregated_context": "Patricia is a librarian who helped you get a library card. They explained the requirements and processed your application. This professional interaction established your relationship with the library staff.",
        "ar_description": "Helped you get library card last Thursday and showed you the mystery section"
    },
    {
        "name": "Patricia",
        "relationship": "Librarian",
        "aggregated_context": "Patricia is a librarian who helped you get a library card and showed you the mystery section. They remembered your interest in mystery books and recommended a new John Grisham title. This shows they pay attention to regular patrons' reading preferences and provide personalized recommendations.",
        "ar_description": "Recommended new John Grisham thriller yesterday, said it just arrived at the library"
    },

    # Ex-colleague examples
    {
        "name": "James",
        "relationship": "Former colleague",
        "aggregated_context": "James is a former colleague you haven't seen in years. You worked together on the Phoenix project during your career. They reminisced about making a great team and facing challenges together. The reunion was warm and brought back positive professional memories.",
        "ar_description": "Ran into each other at coffee shop, reminisced about Phoenix project from 20 years ago"
    },
    {
        "name": "James",
        "relationship": "Former colleague",
        "aggregated_context": "James is a long-time friend from college who shares your love of reading, especially mystery novels. They shared that they retired from IBM last year and are adjusting to retirement life. You congratulated them and acknowledged their long career. This shows ongoing mutual respect from your professional relationship that has transitioned into post-career friendship.",
        "ar_description": "Met for lunch Friday, celebrated his retirement from IBM after 35 years there"
    },

    # Physical therapist
    {
        "name": "David",
        "relationship": "Your physical therapist",
        "aggregated_context": "David is your physical therapist who will be helping with knee recovery. They explained treatment typically takes 8-12 weeks depending on progress. This initial assessment established the treatment plan and timeline.",
        "ar_description": "Started knee rehab sessions Monday, meeting twice weekly for next 8-12 weeks"
    },
    {
        "name": "David",
        "relationship": "Your physical therapist",
        "aggregated_context": "David is your physical therapist helping with knee recovery over 8-12 weeks. They praised your progress with the physical therapy exercises and noted that your range of motion has improved significantly. They recognized your dedication to practicing at home. This shows positive treatment outcomes and an encouraging therapeutic relationship.",
        "ar_description": "Celebrated improved knee flexibility yesterday, said home exercises are really paying off well"
    },

    # Restaurant server
    {
        "name": "Lisa",
        "relationship": "Restaurant server",
        "aggregated_context": "Lisa is a restaurant server at your favorite Italian restaurant who you visited for dinner. They recognized you as a regular customer and remembered your usual order of chicken parmesan. This shows attentiveness to repeat customers and good customer service memory.",
        "ar_description": "Served you at Mario's Italian Restaurant Tuesday, remembered you love chicken parmesan"
    },

    # Pharmacist
    {
        "name": "Richard",
        "relationship": "Your pharmacist",
        "aggregated_context": "Richard is a pharmacist who provided your prescription and carefully reviewed the medication instructions. They specified to take it twice daily with food and followed up on how your new medication is working. You reported improvement with no side effects. This shows their commitment to monitoring patient outcomes beyond just dispensing medications.",
        "ar_description": "Checked on new blood pressure medication Friday, asked about side effects and effectiveness"
    },

    # Hairdresser
    {
        "name": "Nancy",
        "relationship": "Your hairdresser",
        "aggregated_context": "Nancy is a hairdresser who you visited for a trim. They remembered your usual haircut preference and noticed your hair is looking healthier. You mentioned following their product recommendation for the special shampoo. This shows they're attentive to your hair health and you trust their professional advice.",
        "ar_description": "Did your haircut Thursday, noticed hair looks healthier from using recommended shampoo"
    },

    # Grocery clerk
    {
        "name": "Thomas",
        "relationship": "Grocery store worker",
        "aggregated_context": "Thomas is a store employee who works in the produce department. They proactively informed you about fresh strawberries that just arrived and mentioned they're on sale at best price of the season. This shows they recognize you as a regular customer and look out for good deals for you.",
        "ar_description": "Told you about fresh strawberries on sale yesterday, said they're the best of the season"
    },

    # Mechanic
    {
        "name": "Charles",
        "relationship": "Your mechanic",
        "aggregated_context": "Charles is a mechanic who you brought your car to because of a strange grinding noise. They completed the repair of your brake pads and charged less than originally quoted at $175. This honest pricing builds trust and shows integrity in their business practices.",
        "ar_description": "Fixed car's grinding brake noise Monday, charged less than quoted at only $175"
    },

    # Yoga instructor
    {
        "name": "Linda",
        "relationship": "Yoga instructor",
        "aggregated_context": "Linda is a yoga instructor who welcomed you to your first class. They were reassuring about going at your own pace and offered to provide modifications as needed. They noticed your improved balance and acknowledged your practice efforts. They suggested progressing to more advanced poses.",
        "ar_description": "Praised balance improvement at yoga class Wednesday, suggested trying tree pose next week"
    },

    # Personal trainer
    {
        "name": "Christopher",
        "relationship": "Your personal trainer",
        "aggregated_context": "Christopher is your personal trainer who discussed your fitness goals to improve strength. They committed to creating a personalized training plan and observed significant strength improvements since you began training together. They suggested increasing workout intensity by adding more weight.",
        "ar_description": "Increased workout weights Tuesday after seeing strength gains, doing great with training plan"
    },

    # Taxi driver
    {
        "name": "Daniel",
        "relationship": "Taxi driver",
        "aggregated_context": "Daniel is a taxi driver who picked you up for a ride to the medical center. They recognized you as a regular passenger and remembered your typical destination. They also showed personal interest by asking how you've been.",
        "ar_description": "Drove you to medical center appointment Friday, asked how physical therapy is going"
    },
]


def generate_training_dataset(num_examples: int = 1000) -> List[dict]:
    """Generate training dataset by expanding and varying the base examples."""
    training_data = []

    # Use base examples multiple times with variations
    for _ in range(num_examples):
        # Pick a random base example
        base = random.choice(TRAINING_EXAMPLES)

        # Create training example in Fireworks format
        example = {
            "messages": [
                {
                    "role": "system",
                    "content": AR_DISPLAY_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"Person Name: {base['name']}\nRelationship: {base['relationship']}\n\nAggregated Context:\n{base['aggregated_context']}"
                },
                {
                    "role": "assistant",
                    "content": base['ar_description']
                }
            ]
        }

        training_data.append(example)

    # Shuffle to mix different relationship types
    random.shuffle(training_data)

    return training_data


def save_to_jsonl(data: List[dict], filename: str):
    """Save training data to JSONL format."""
    with open(filename, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    print(f"Saved {len(data)} examples to {filename}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    print("Generating AR Display Model training dataset...")
    print(f"Base examples: {len(TRAINING_EXAMPLES)}")

    # Generate 1000 examples by sampling from base examples
    training_examples = generate_training_dataset(1000)

    # Save to JSONL
    output_path = "data/ar_display_training.jsonl"
    save_to_jsonl(training_examples, output_path)

    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLES:")
    print("="*80)

    # Show 3 sample examples
    for i in range(min(3, len(training_examples))):
        print(f"\nExample {i+1}:")
        print(json.dumps(training_examples[i], indent=2))
        print("-"*80)

    print("\n" + "="*80)
    print("DATASET SUMMARY:")
    print("="*80)
    print(f"Total examples: {len(training_examples)}")
    print(f"Relationship types covered: {len(set(ex['relationship'] for ex in TRAINING_EXAMPLES))}")
    print("\nRelationship distribution in base examples:")
    relationships = [ex['relationship'] for ex in TRAINING_EXAMPLES]
    for rel in set(relationships):
        count = relationships.count(rel)
        print(f"  - {rel}: {count} examples")
