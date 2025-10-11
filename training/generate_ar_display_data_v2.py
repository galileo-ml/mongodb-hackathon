"""Generate synthetic training data for AR Display Model (Model 2) - EXPANDED VERSION.

This model takes:
- Input: Person name, relationship, verbose aggregated context
- Output: One-line specific description for AR display (15-20 words)

EXPANDED with:
- Grandchildren with specific interests and activities
- Friends with hobbies, jobs, and life updates
- Diverse professional relationships
- Specific events, places, and memorable details
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


# Extensive training examples with diverse scenarios
TRAINING_EXAMPLES = [
    # ===== GRANDCHILDREN - Specific activities, hobbies, school =====
    {
        "name": "Emma",
        "relationship": "Your granddaughter",
        "aggregated_context": "Emma is your 12-year-old granddaughter who loves playing soccer. She visited last weekend and was very excited about making the varsity team at her middle school. She practiced a new move called a rainbow kick and demonstrated it for you in the backyard. She mentioned her first tournament is next month in Sacramento. You encouraged her and she promised to tell you all about it.",
        "ar_description": "Visited Sunday showing off soccer skills, made varsity team with tournament in Sacramento next month"
    },
    {
        "name": "Noah",
        "relationship": "Your grandson",
        "aggregated_context": "Noah is your 10-year-old grandson who is passionate about building robots. He came over Tuesday after school to show you his latest LEGO robot that can move and pick up objects. He explained how he programmed it using a tablet app. He's entering it in the county science fair next week and is very nervous but excited. You told him you're proud of his creativity.",
        "ar_description": "Showed off LEGO robot Tuesday, entering county science fair next week with moving creation"
    },
    {
        "name": "Olivia",
        "relationship": "Your granddaughter",
        "aggregated_context": "Olivia is your 8-year-old granddaughter who loves art and painting. She brought over her watercolor paintings from art class on Friday and gave you one she painted of your house with flowers. She said she's learning about impressionism from her teacher Mrs. Peterson. She asked if you could come to her art show at school next Thursday evening.",
        "ar_description": "Gave you watercolor painting of your house Friday, invited you to school art show next Thursday"
    },
    {
        "name": "Liam",
        "relationship": "Your grandson",
        "aggregated_context": "Liam is your 14-year-old grandson who plays guitar in a band with his friends. He visited yesterday and played his new song for you that his band wrote about summer vacation. The band is called The Thunderbirds and they're performing at the school talent show on Friday night. He invited you to come watch and said they might win first place.",
        "ar_description": "Played new guitar song yesterday, band The Thunderbirds performing at school talent show Friday"
    },
    {
        "name": "Sophia",
        "relationship": "Your granddaughter",
        "aggregated_context": "Sophia is your 16-year-old granddaughter who just got her driver's license. She came over Saturday morning in her mom's car to show you that she can drive now. She's been practicing parallel parking and is getting really good at it. She offered to drive you to your doctor's appointment next week, which made her feel very grown up and responsible.",
        "ar_description": "Got driver's license last week, drove over Saturday and wants to drive you to appointments"
    },
    {
        "name": "Mason",
        "relationship": "Your grandson",
        "aggregated_context": "Mason is your 11-year-old grandson who is obsessed with dinosaurs and paleontology. He visited Thursday after a field trip to the Natural History Museum in San Francisco. He couldn't stop talking about seeing the Tyrannosaurus Rex skeleton and learning about the Jurassic period. He wants to be a paleontologist when he grows up and asked if you remember when dinosaurs were discovered.",
        "ar_description": "Visited Thursday after museum field trip, excited about T-Rex and wants to be paleontologist"
    },
    {
        "name": "Ava",
        "relationship": "Your granddaughter",
        "aggregated_context": "Ava is your 13-year-old granddaughter who is learning to bake and loves making desserts. She brought over homemade chocolate chip cookies on Monday that she baked herself for the first time. The cookies were delicious and she was very proud. She's planning to bake a birthday cake for her friend's party next Saturday and asked for your advice on decorating.",
        "ar_description": "Brought homemade chocolate chip cookies Monday, planning to bake friend's birthday cake next Saturday"
    },
    {
        "name": "Lucas",
        "relationship": "Your grandson",
        "aggregated_context": "Lucas is your 9-year-old grandson who loves reading adventure books. He finished the entire Harry Potter series and came over Wednesday to tell you all about it. His favorite character is Hermione because she's smart and brave. He's now starting the Percy Jackson series and is very excited. He asked if you like to read adventure stories too.",
        "ar_description": "Visited Wednesday excited about finishing Harry Potter series, starting Percy Jackson books next"
    },

    # ===== FRIENDS - Hobbies, interests, life updates =====
    {
        "name": "Margaret",
        "relationship": "Your friend",
        "aggregated_context": "Margaret is your friend from the community garden who you've known for 15 years. She came by on Tuesday with fresh tomatoes from her garden and talked about entering the county fair's tomato competition next month. She's been growing a special heirloom variety called Cherokee Purple. She invited you to come with her to the fair and said you could enter your roses too.",
        "ar_description": "Brought Cherokee Purple tomatoes Tuesday, entering county fair garden competition next month together"
    },
    {
        "name": "William",
        "relationship": "Your friend",
        "aggregated_context": "William is your friend who shares your passion for woodworking. He stopped by Saturday morning to show you the oak rocking chair he just finished building in his workshop. It took him three months to complete and he hand-carved details on the armrests. He's thinking about selling his furniture at the craft fair downtown next weekend and asked for your opinion on pricing.",
        "ar_description": "Showed oak rocking chair Saturday, hand-carved details, selling furniture at craft fair next weekend"
    },
    {
        "name": "Dorothy",
        "relationship": "Your friend",
        "aggregated_context": "Dorothy is your friend from the senior center's painting class. She called Thursday evening to tell you about the painting workshop she signed both of you up for next Tuesday at the community center. The instructor is teaching watercolor landscapes and the theme is California coastlines. She's very excited and asked if you could carpool together at 10am.",
        "ar_description": "Signed you up for watercolor landscape class next Tuesday, painting California coastlines together"
    },
    {
        "name": "George",
        "relationship": "Your friend",
        "aggregated_context": "George is your friend and chess partner who you play with every Friday afternoon at the park. Last Friday he finally beat you after losing for three weeks straight. He used a new opening strategy he learned from a YouTube video. He's been studying chess tactics and wants to enter a local tournament at the library next month. He asked if you'd come watch and support him.",
        "ar_description": "Finally won at chess Friday using new strategy, entering library tournament next month"
    },
    {
        "name": "Barbara",
        "relationship": "Your friend",
        "aggregated_context": "Barbara is your friend who loves bird watching and photography. She came over Monday with beautiful photos she took at the wetlands preserve of a great blue heron catching fish. She's been going early in the mornings to capture the best light. She mentioned the spring bird migration starts next week and invited you to join her on a bird watching trip to Point Reyes.",
        "ar_description": "Showed heron photos from wetlands Monday, planning Point Reyes bird watching trip next week"
    },
    {
        "name": "Richard",
        "relationship": "Your friend",
        "aggregated_context": "Richard is your friend who recently started taking ballroom dancing lessons with his wife. He stopped by Wednesday to tell you all about learning the foxtrot and tango. They have a dance recital coming up at the community center on Saturday evening and he personally invited you to attend. He's nervous but excited and said they've been practicing for two months.",
        "ar_description": "Learning ballroom dancing with wife, performing foxtrot at community center recital this Saturday"
    },
    {
        "name": "Helen",
        "relationship": "Your friend",
        "aggregated_context": "Helen is your friend who volunteers at the animal shelter and is passionate about rescue dogs. She visited Friday afternoon with photos of a golden retriever puppy she's fostering named Buddy. The puppy is 8 weeks old and needs a forever home. She's organizing an adoption event at the shelter this Sunday and asked if you might know anyone interested in adopting a dog.",
        "ar_description": "Fostering golden retriever puppy Buddy, hosting shelter adoption event this Sunday afternoon"
    },
    {
        "name": "Thomas",
        "relationship": "Your friend",
        "aggregated_context": "Thomas is your friend who is learning to play piano at age 72. He came over Monday beaming with pride because he finally learned to play Beethoven's Moonlight Sonata after practicing for six months. He played it for you on your piano and you were very impressed. His piano teacher is having a student recital next month at the music academy and he's performing.",
        "ar_description": "Played Beethoven's Moonlight Sonata Monday after 6 months practice, recital at music academy next month"
    },

    # ===== FRIENDS - Career updates and job pursuits =====
    {
        "name": "Jennifer",
        "relationship": "Your friend",
        "aggregated_context": "Jennifer is your friend who recently went back to school to become a nurse at age 55. She stopped by Tuesday after her first clinical rotation at the hospital. She was exhausted but excited about working in the pediatric ward. She has exams coming up next week and will graduate from nursing school in three months. You told her how proud you are of her courage to start a new career.",
        "ar_description": "Finished first clinical rotation Tuesday in pediatric ward, graduates nursing school in three months"
    },
    {
        "name": "David",
        "relationship": "Your friend",
        "aggregated_context": "David is your friend who just got promoted to regional manager at his logistics company. He came over Saturday to celebrate with you and brought champagne. The promotion means he'll be overseeing five warehouses across Northern California. He starts his new position next Monday and is both excited and nervous about managing a bigger team. You congratulated him on his 20 years of hard work paying off.",
        "ar_description": "Promoted to regional manager Saturday, overseeing five warehouses starting next Monday after 20 years"
    },
    {
        "name": "Patricia",
        "relationship": "Your friend",
        "aggregated_context": "Patricia is your friend who started her own bakery business from home six months ago. She brought over fresh sourdough bread and croissants on Friday that she baked at 4am. Her specialty is French pastries and her business has grown so much through Instagram that she's opening a storefront in downtown next month. She invited you to the grand opening celebration.",
        "ar_description": "Brought fresh sourdough Friday, opening French bakery storefront downtown next month after Instagram success"
    },
    {
        "name": "Robert",
        "relationship": "Your friend",
        "aggregated_context": "Robert is your friend who recently published his first mystery novel at age 68 after writing as a hobby for decades. He came over Wednesday with an autographed copy of his book called 'The Midnight Detective.' The book launch event is this Saturday at the local bookstore where he'll be doing a reading and signing copies. He's incredibly nervous about speaking in public but thrilled to finally be a published author.",
        "ar_description": "Published first mystery novel 'The Midnight Detective', book launch signing at bookstore this Saturday"
    },
    {
        "name": "Linda",
        "relationship": "Your friend",
        "aggregated_context": "Linda is your friend who started teaching yoga classes at the community center after getting certified last year. She stopped by Monday to invite you to her new gentle yoga class specifically designed for seniors. The class meets Tuesday and Thursday mornings at 9am and focuses on flexibility and balance. She said the first class is free and she'd love for you to try it.",
        "ar_description": "Started senior yoga classes at community center, invited you to free trial Tuesday morning"
    },
    {
        "name": "Charles",
        "relationship": "Your friend",
        "aggregated_context": "Charles is your friend who recently retired from teaching high school math and is now tutoring students part-time. He came over Thursday excited about helping a struggling student finally understand algebra. The student got an A on their test after weeks of tutoring sessions. He finds retirement tutoring more rewarding than his full career. He's starting a free math help program at the library next month.",
        "ar_description": "Student got A on algebra test Thursday, starting free math tutoring program at library next month"
    },

    # ===== FAMILY - Adult children with specific careers =====
    {
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Sarah is your daughter who works as a software engineer at Google in Mountain View. She visited last Sunday and told you about the artificial intelligence project she's leading that will help doctors diagnose diseases earlier. Her team just got approval for a $2 million budget and she's hiring five new engineers. She's been working late nights but loves the challenge and impact of the work.",
        "ar_description": "Visited Sunday excited about leading AI medical diagnosis project at Google with $2M budget"
    },
    {
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Michael is your son who is a high school history teacher in Oakland. He came over Wednesday evening to tell you about his students' Civil War history project presentations. One student's project on the Underground Railroad was so good that it's being entered in the state competition. He's planning a field trip to Gettysburg next month and has been fundraising to help students who can't afford the trip.",
        "ar_description": "Student won state history competition Wednesday, planning Gettysburg field trip next month with fundraising"
    },
    {
        "name": "Jessica",
        "relationship": "Your daughter",
        "aggregated_context": "Jessica is your daughter who works as a veterinarian at the Oakland Zoo. She stopped by Saturday morning with photos of a baby elephant that was born at the zoo on Thursday. She helped deliver the elephant and the whole experience was emotional and amazing. The zoo is having a naming contest for the baby elephant and she invited you to come meet the new arrival next weekend.",
        "ar_description": "Helped deliver baby elephant at zoo Thursday, invited you to visit and name contest next weekend"
    },
    {
        "name": "Daniel",
        "relationship": "Your son",
        "aggregated_context": "Daniel is your son who owns a restaurant in San Francisco called The Blue Heron. He came over Tuesday after a food critic from the San Francisco Chronicle reviewed his restaurant and gave it 4 stars. The review praised his signature dish, wild salmon with lemon butter sauce. He's been getting reservation requests flooding in and might need to hire more staff. He wants you to come for dinner this Friday to celebrate.",
        "ar_description": "Restaurant got 4-star Chronicle review Tuesday for salmon dish, wants to celebrate Friday dinner"
    },

    # ===== ORIGINAL DIVERSE EXAMPLES FROM V1 =====
    {
        "name": "Sarah",
        "relationship": "Your daughter",
        "aggregated_context": "Sarah is your daughter. She just shared exciting news about receiving a promotion at work. She mentioned that the promotion came after months of hard work on a major project called the Phoenix project. She seemed very happy and excited about this achievement. The conversation shows a close, supportive relationship where she regularly checks in on your wellbeing.",
        "ar_description": "Visited 3 days ago celebrating her promotion after completing the Phoenix project at work"
    },
    {
        "name": "Michael",
        "relationship": "Your son",
        "aggregated_context": "Michael is your son. He brought groceries for you, including your favorite sourdough bread and fresh fruit. This shows he pays attention to your preferences and takes care of your practical needs. He checked in on how you've been managing, demonstrating regular concern for your wellbeing. The relationship appears caring and supportive, with him being described as 'always thoughtful.'",
        "ar_description": "Dropped by yesterday with groceries including your favorite sourdough bread and fresh strawberries"
    },
    {
        "name": "Robert",
        "relationship": "Your friend from book club",
        "aggregated_context": "Robert is your friend from book club. You're both reading a mystery novel that you find quite good and are almost finished with. Book club meets next Tuesday and will be discussing the ending of the book. Robert reminded you to finish the book before the meeting, and you confirmed you'll complete it this weekend. The conversation indicates a shared intellectual interest and regular social engagement through the book club.",
        "ar_description": "Book club meets Tuesday to discuss the new Agatha Christie mystery you're both reading"
    },
    {
        "name": "Jennifer",
        "relationship": "Your neighbor",
        "aggregated_context": "Jennifer is your new neighbor who just moved in next door to apartment 3B. They introduced themselves and offered to help if you ever need anything. The interaction was friendly and welcoming, establishing a positive neighbor relationship from the start.",
        "ar_description": "Just moved into apartment 3B last week and offered to help with anything you need"
    },
    {
        "name": "Maria",
        "relationship": "Your caregiver",
        "aggregated_context": "Maria is your caregiver who will be assisting with medication management three times a week. This was their introduction meeting where they explained their role. They offered to answer any questions, showing a patient-centered approach to care.",
        "ar_description": "Started as new caregiver Monday, helps with medication three times per week"
    },
    {
        "name": "Dr. Anderson",
        "relationship": "Your doctor",
        "aggregated_context": "Dr. Anderson is your new primary care physician. During the first appointment, they reviewed your medical history and discussed your current medications. They demonstrated thoroughness in reviewing your health information before treatment.",
        "ar_description": "First appointment last Monday, reviewed medical history and adjusted blood pressure medication"
    },
    {
        "name": "Patricia",
        "relationship": "Librarian",
        "aggregated_context": "Patricia is a librarian who helped you get a library card and showed you the mystery section. They remembered your interest in mystery books and recommended a new John Grisham title. This shows they pay attention to regular patrons' reading preferences and provide personalized recommendations.",
        "ar_description": "Recommended new John Grisham thriller yesterday, said it just arrived at the library"
    },
    {
        "name": "James",
        "relationship": "Former colleague",
        "aggregated_context": "James is a former colleague you haven't seen in years. You worked together on the Phoenix project during your career. They reminisced about making a great team and facing challenges together. The reunion was warm and brought back positive professional memories.",
        "ar_description": "Ran into each other at coffee shop, reminisced about Phoenix project from 20 years ago"
    },
    {
        "name": "David",
        "relationship": "Your physical therapist",
        "aggregated_context": "David is your physical therapist helping with knee recovery over 8-12 weeks. They praised your progress with the physical therapy exercises and noted that your range of motion has improved significantly. They recognized your dedication to practicing at home. This shows positive treatment outcomes and an encouraging therapeutic relationship.",
        "ar_description": "Celebrated improved knee flexibility yesterday, said home exercises are really paying off well"
    },
    {
        "name": "Lisa",
        "relationship": "Restaurant server",
        "aggregated_context": "Lisa is a restaurant server at your favorite Italian restaurant who you visited for dinner. They recognized you as a regular customer and remembered your usual order of chicken parmesan. This shows attentiveness to repeat customers and good customer service memory.",
        "ar_description": "Served you at Mario's Italian Restaurant Tuesday, remembered you love chicken parmesan"
    },
    {
        "name": "Richard",
        "relationship": "Your pharmacist",
        "aggregated_context": "Richard is a pharmacist who provided your prescription and carefully reviewed the medication instructions. They specified to take it twice daily with food and followed up on how your new medication is working. You reported improvement with no side effects. This shows their commitment to monitoring patient outcomes beyond just dispensing medications.",
        "ar_description": "Checked on new blood pressure medication Friday, asked about side effects and effectiveness"
    },
    {
        "name": "Nancy",
        "relationship": "Your hairdresser",
        "aggregated_context": "Nancy is a hairdresser who you visited for a trim. They remembered your usual haircut preference and noticed your hair is looking healthier. You mentioned following their product recommendation for the special shampoo. This shows they're attentive to your hair health and you trust their professional advice.",
        "ar_description": "Did your haircut Thursday, noticed hair looks healthier from using recommended shampoo"
    },
    {
        "name": "Thomas",
        "relationship": "Grocery store worker",
        "aggregated_context": "Thomas is a store employee who works in the produce department. They proactively informed you about fresh strawberries that just arrived and mentioned they're on sale at best price of the season. This shows they recognize you as a regular customer and look out for good deals for you.",
        "ar_description": "Told you about fresh strawberries on sale yesterday, said they're the best of the season"
    },
    {
        "name": "Charles",
        "relationship": "Your mechanic",
        "aggregated_context": "Charles is a mechanic who you brought your car to because of a strange grinding noise. They completed the repair of your brake pads and charged less than originally quoted at $175. This honest pricing builds trust and shows integrity in their business practices.",
        "ar_description": "Fixed car's grinding brake noise Monday, charged less than quoted at only $175"
    },
    {
        "name": "Linda",
        "relationship": "Yoga instructor",
        "aggregated_context": "Linda is a yoga instructor who welcomed you to your first class. They were reassuring about going at your own pace and offered to provide modifications as needed. They noticed your improved balance and acknowledged your practice efforts. They suggested progressing to more advanced poses.",
        "ar_description": "Praised balance improvement at yoga class Wednesday, suggested trying tree pose next week"
    },
    {
        "name": "Christopher",
        "relationship": "Your personal trainer",
        "aggregated_context": "Christopher is your personal trainer who discussed your fitness goals to improve strength. They committed to creating a personalized training plan and observed significant strength improvements since you began training together. They suggested increasing workout intensity by adding more weight.",
        "ar_description": "Increased workout weights Tuesday after seeing strength gains, doing great with training plan"
    },
    {
        "name": "Daniel",
        "relationship": "Taxi driver",
        "aggregated_context": "Daniel is a taxi driver who picked you up for a ride to the medical center. They recognized you as a regular passenger and remembered your typical destination. They also showed personal interest by asking how you've been.",
        "ar_description": "Drove you to medical center appointment Friday, asked how physical therapy is going"
    },
]


def generate_training_dataset(num_examples: int = 1000) -> List[dict]:
    """Generate training dataset by sampling from diverse base examples."""
    training_data = []

    # Use base examples multiple times
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

    print("Generating EXPANDED AR Display Model training dataset...")
    print(f"Base examples: {len(TRAINING_EXAMPLES)}")

    # Count categories
    grandchildren = len([ex for ex in TRAINING_EXAMPLES if 'grandson' in ex['relationship'] or 'granddaughter' in ex['relationship']])
    friends = len([ex for ex in TRAINING_EXAMPLES if 'friend' in ex['relationship'].lower()])
    family = len([ex for ex in TRAINING_EXAMPLES if 'daughter' in ex['relationship'] or 'son' in ex['relationship']])

    print(f"  - Grandchildren examples: {grandchildren}")
    print(f"  - Friends with hobbies/careers: {friends}")
    print(f"  - Adult children with careers: {family}")
    print(f"  - Other relationships: {len(TRAINING_EXAMPLES) - grandchildren - friends - family}")

    # Generate 1000 examples by sampling from base examples
    training_examples = generate_training_dataset(1000)

    # Save to JSONL
    output_path = "data/ar_display_training.jsonl"
    save_to_jsonl(training_examples, output_path)

    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLES:")
    print("="*80)

    # Show diverse samples
    sample_indices = [0, 10, 20, 30, 40]
    for idx in sample_indices:
        if idx < len(training_examples):
            print(f"\nExample {idx+1}:")
            print(json.dumps(training_examples[idx], indent=2))
            print("-"*80)

    print("\n" + "="*80)
    print("DATASET SUMMARY:")
    print("="*80)
    print(f"Total examples: {len(training_examples)}")
    print(f"Unique base scenarios: {len(TRAINING_EXAMPLES)}")
