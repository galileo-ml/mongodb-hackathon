"""Generate synthetic training data for Context Aggregation Model (Model 1).

This model takes Previous Context + Current Conversation and outputs Detailed Updated Context.
The output is verbose (3-5 sentences) for MongoDB storage, not for AR display.
"""

import json
import random
from typing import List, Dict


# Conversation templates for different topics and relationships
CONVERSATION_TEMPLATES = {
    "daughter": {
        "first_meeting_topics": [
            {
                "topic": "work_promotion",
                "conversation": [
                    ("person", "Hi dad, how are you feeling today?"),
                    ("patient", "I'm doing well, thanks for asking."),
                    ("person", "I got that promotion at work I mentioned!"),
                    ("patient", "That's wonderful news! Congratulations!"),
                    ("person", "It came after months of hard work on the {project} project."),
                    ("patient", "I'm so proud of you."),
                ],
                "context_output": "{name} is your daughter. She just shared exciting news about receiving a promotion at work. She mentioned that the promotion came after months of hard work on a major project called the {project} project. She seemed very happy and excited about this achievement. The conversation shows a close, supportive relationship where she regularly checks in on your wellbeing."
            },
            {
                "topic": "health_concern",
                "conversation": [
                    ("person", "How was your doctor's appointment yesterday?"),
                    ("patient", "It went well, everything looks good."),
                    ("person", "That's a relief! Did they adjust any medications?"),
                    ("patient", "No changes needed."),
                    ("person", "Great! I was worried about your {health_issue}."),
                    ("patient", "The doctor said it's under control now."),
                ],
                "context_output": "{name} is your daughter who shows deep concern for your health and wellbeing. She asked about your recent doctor's appointment and was particularly concerned about your {health_issue}. You reported that the appointment went well with no medication changes needed and your {health_issue} is now under control. Her relief at the good news demonstrates her ongoing worry and care for your health."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "kids_visiting",
                "conversation": [
                    ("person", "The kids are so excited to visit you next weekend."),
                    ("patient", "I can't wait to see them."),
                    ("person", "They've been asking about you non-stop."),
                    ("patient", "That's so sweet."),
                ],
                "context_output": "She mentioned that her children (your grandchildren) are planning to visit next weekend. The children have been asking about you frequently, showing excitement about the upcoming visit. The family maintains close bonds with regular visits and ongoing communication."
            },
        ]
    },
    "son": {
        "first_meeting_topics": [
            {
                "topic": "groceries",
                "conversation": [
                    ("person", "Hey dad, I brought some groceries for you."),
                    ("patient", "Thank you son, you're always so thoughtful."),
                    ("person", "I got your favorite {food_item} and some fresh fruit."),
                    ("patient", "That's perfect."),
                ],
                "context_output": "{name} is your son. He brought groceries for you, including your favorite {food_item} and fresh fruit. This shows he pays attention to your preferences and takes care of your practical needs. The relationship appears caring and supportive, with him being described as 'always thoughtful.'"
            },
        ],
        "follow_up_topics": [
            {
                "topic": "camping_trip",
                "conversation": [
                    ("person", "I'm planning a camping trip next month."),
                    ("patient", "Sounds like fun, be safe out there."),
                    ("person", "It's up in the {location}, really beautiful area."),
                    ("patient", "Take lots of photos."),
                ],
                "context_output": "He shared his plans for a camping trip next month in the {location}. He mentioned it's a beautiful area and promised to take photos to share with you. The relationship shows mutual care and regular communication about life updates."
            },
        ]
    },
    "friend": {
        "first_meeting_topics": [
            {
                "topic": "book_club",
                "conversation": [
                    ("person", "Have you finished that {genre} novel yet?"),
                    ("patient", "Almost done, it's quite good."),
                    ("person", "Book club is meeting next {day}."),
                    ("patient", "I'll be there."),
                ],
                "context_output": "{name} is your friend from book club. You're both reading a {genre} novel that you find quite good. Book club meets next {day} and will be discussing the ending. The conversation indicates a shared intellectual interest and regular social engagement."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "college_memories",
                "conversation": [
                    ("person", "Remember when we used to {activity} in college?"),
                    ("patient", "Those were good times."),
                    ("person", "I found an old photo of us from graduation."),
                    ("patient", "I'd love to see that."),
                ],
                "context_output": "He reminisced about {activity} together in college and mentioned finding an old graduation photo of you both. The friendship spans decades and includes fond memories from your college years."
            },
        ]
    },
    "neighbor": {
        "first_meeting_topics": [
            {
                "topic": "introduction",
                "conversation": [
                    ("person", "Hi, I'm {name}, I just moved in next door."),
                    ("patient", "Welcome to the neighborhood!"),
                    ("person", "Thanks! Let me know if you ever need anything."),
                    ("patient", "That's very kind of you."),
                    ("person", "I'm in apartment {number}."),
                    ("patient", "Good to know."),
                ],
                "context_output": "{name} is your new neighbor who just moved in next door to apartment {number}. They introduced themselves and offered to help if you ever need anything. The interaction was friendly and welcoming, establishing a positive neighbor relationship from the start."
            },
            {
                "topic": "package_delivery",
                "conversation": [
                    ("person", "A package was delivered to my door for you by mistake."),
                    ("patient", "Oh, thank you for bringing it over."),
                    ("person", "No problem, happens all the time."),
                    ("patient", "I appreciate it."),
                ],
                "context_output": "{name} is your neighbor who brought over a package that was mistakenly delivered to their address. They were helpful and mentioned this happens frequently. This shows they're attentive and willing to assist with small neighborly favors."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "noise_apology",
                "conversation": [
                    ("person", "Sorry about the noise last night, we had guests over."),
                    ("patient", "No worries, I barely noticed."),
                    ("person", "I just wanted to make sure it wasn't too disruptive."),
                    ("patient", "You're very considerate."),
                ],
                "context_output": "They apologized for noise from having guests over last night, showing consideration for your comfort. You reassured them it wasn't disruptive. This demonstrates a respectful and considerate neighbor dynamic."
            },
        ]
    },
    "caregiver": {
        "first_meeting_topics": [
            {
                "topic": "introduction",
                "conversation": [
                    ("person", "Hello, I'm {name}, I'll be your caregiver."),
                    ("patient", "Nice to meet you."),
                    ("person", "I'll be helping with {care_task} three times a week."),
                    ("patient", "That sounds good."),
                    ("person", "Do you have any questions for me?"),
                    ("patient", "Not right now, thank you."),
                ],
                "context_output": "{name} is your caregiver who will be assisting with {care_task} three times a week. This was their introduction meeting where they explained their role. They offered to answer any questions, showing a patient-centered approach to care."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "medication_reminder",
                "conversation": [
                    ("person", "Time for your {time} medication."),
                    ("patient", "Thank you for reminding me."),
                    ("person", "Have you eaten anything yet today?"),
                    ("patient", "Yes, I had breakfast."),
                ],
                "context_output": "They provided a medication reminder for your {time} dose and checked whether you had eaten, showing attention to proper medication administration. This demonstrates their diligent and caring approach to their caregiver responsibilities."
            },
        ]
    },
    "ex_colleague": {
        "first_meeting_topics": [
            {
                "topic": "reunion",
                "conversation": [
                    ("person", "It's been years! How have you been?"),
                    ("patient", "Good to see you! I've been well."),
                    ("person", "I remember working together on the {project} project."),
                    ("patient", "Those were challenging times."),
                    ("person", "We made a great team back then."),
                    ("patient", "We certainly did."),
                ],
                "context_output": "{name} is a former colleague you haven't seen in years. You worked together on the {project} project during your career. They reminisced about making a great team and facing challenges together. The reunion was warm and brought back positive professional memories."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "career_update",
                "conversation": [
                    ("person", "I retired from {company} last year."),
                    ("patient", "Congratulations! How's retirement treating you?"),
                    ("person", "It's an adjustment, but I'm enjoying it."),
                    ("patient", "You earned it after all those years."),
                ],
                "context_output": "They shared that they retired from {company} last year and are adjusting to retirement life. You congratulated them and acknowledged their long career. This shows ongoing mutual respect from your professional relationship that has transitioned into post-career friendship."
            },
        ]
    },
    "postal_worker": {
        "first_meeting_topics": [
            {
                "topic": "regular_route",
                "conversation": [
                    ("person", "Hi, I'm {name}, your regular mail carrier now."),
                    ("patient", "Nice to meet you."),
                    ("person", "I'll be delivering on this route from now on."),
                    ("patient", "Good to know."),
                    ("person", "If you need anything signed, just leave a note."),
                    ("patient", "I appreciate that."),
                ],
                "context_output": "{name} is your new regular mail carrier who introduced themselves. They explained they'll be on this route going forward and provided helpful information about signing for packages. This professional introduction establishes a working relationship for regular mail delivery."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "package_handling",
                "conversation": [
                    ("person", "I have a large package for you today."),
                    ("patient", "Oh, thank you!"),
                    ("person", "I'll leave it inside the door like usual."),
                    ("patient", "That's very helpful."),
                ],
                "context_output": "They delivered a large package and offered to place it inside your door as usual, showing familiarity with your preferences. This indicates a considerate working relationship that has developed over regular interactions."
            },
        ]
    },
    "grocery_clerk": {
        "first_meeting_topics": [
            {
                "topic": "store_introduction",
                "conversation": [
                    ("person", "Welcome! I'm {name}, I work in the {department} department."),
                    ("patient", "Nice to meet you."),
                    ("person", "Let me know if you need help finding anything."),
                    ("patient", "Thank you, I will."),
                ],
                "context_output": "{name} is a store employee who works in the {department} department. They introduced themselves and offered assistance with finding items. This professional interaction shows their customer service oriented approach."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "product_recommendation",
                "conversation": [
                    ("person", "We just got fresh {produce_item} in today."),
                    ("patient", "Those look great!"),
                    ("person", "They're on sale too, best price of the season."),
                    ("patient", "I'll take some, thanks for letting me know."),
                ],
                "context_output": "They proactively informed you about fresh {produce_item} that just arrived and mentioned they're on sale. This shows they recognize you as a regular customer and look out for good deals for you."
            },
        ]
    },
    "doctor": {
        "first_meeting_topics": [
            {
                "topic": "new_patient",
                "conversation": [
                    ("person", "Hello, I'm Dr. {name}, I'll be your new primary care physician."),
                    ("patient", "Nice to meet you, Doctor."),
                    ("person", "I've reviewed your medical history."),
                    ("patient", "Okay, good."),
                    ("person", "Let's discuss your current medications."),
                    ("patient", "Sure, I have the list here."),
                ],
                "context_output": "Dr. {name} is your new primary care physician. During the first appointment, they reviewed your medical history and discussed your current medications. They demonstrated thoroughness in reviewing your health information before treatment."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "test_results",
                "conversation": [
                    ("person", "Your recent {test_type} results look good."),
                    ("patient", "That's a relief!"),
                    ("person", "Keep up with your current treatment plan."),
                    ("patient", "I will, thank you."),
                ],
                "context_output": "They reviewed your {test_type} results and reported they look good. They recommended continuing your current treatment plan. This follow-up shows ongoing monitoring of your health conditions."
            },
        ]
    },
    "physical_therapist": {
        "first_meeting_topics": [
            {
                "topic": "initial_assessment",
                "conversation": [
                    ("person", "I'm {name}, your physical therapist."),
                    ("patient", "Hello."),
                    ("person", "We'll be working on your {body_part} recovery."),
                    ("patient", "How long will treatment take?"),
                    ("person", "Usually {duration} weeks, depending on progress."),
                    ("patient", "I understand."),
                ],
                "context_output": "{name} is your physical therapist who will be helping with {body_part} recovery. They explained treatment typically takes {duration} weeks depending on progress. This initial assessment established the treatment plan and timeline."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "progress_check",
                "conversation": [
                    ("person", "You're making great progress with the exercises."),
                    ("patient", "Thank you, I've been practicing at home."),
                    ("person", "It shows! Your {measurement} has improved significantly."),
                    ("patient", "That's encouraging."),
                ],
                "context_output": "They praised your progress with the physical therapy exercises and noted that your {measurement} has improved significantly. They recognized your dedication to practicing at home. This shows positive treatment outcomes and an encouraging therapeutic relationship."
            },
        ]
    },
    "hairdresser": {
        "first_meeting_topics": [
            {
                "topic": "new_client",
                "conversation": [
                    ("person", "Hi! I'm {name}, what are we doing today?"),
                    ("patient", "Just a trim, please."),
                    ("person", "How much would you like taken off?"),
                    ("patient", "About an inch."),
                ],
                "context_output": "{name} is a hairdresser who you visited for a trim. They asked about your preferences for the haircut. This professional service interaction was straightforward and focused on understanding your styling needs."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "regular_appointment",
                "conversation": [
                    ("person", "The usual today?"),
                    ("patient", "Yes, please."),
                    ("person", "Your hair is looking healthier than last time."),
                    ("patient", "I've been using that product you recommended."),
                ],
                "context_output": "They remembered your usual haircut preference and noticed your hair is looking healthier. You mentioned following their product recommendation. This shows they're attentive to your hair health and you trust their professional advice."
            },
        ]
    },
    "librarian": {
        "first_meeting_topics": [
            {
                "topic": "library_card",
                "conversation": [
                    ("person", "Hello, I'm {name}, can I help you with anything today?"),
                    ("patient", "I'd like to get a library card."),
                    ("person", "Great! I'll need to see an ID and proof of address."),
                    ("patient", "I have both right here."),
                ],
                "context_output": "{name} is a librarian who helped you get a library card. They explained the requirements and processed your application. This professional interaction established your relationship with the library staff."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "book_recommendation",
                "conversation": [
                    ("person", "Looking for another {genre} book?"),
                    ("patient", "Yes, do you have any suggestions?"),
                    ("person", "This new {author} just came in, I think you'd enjoy it."),
                    ("patient", "I'll check it out, thanks!"),
                ],
                "context_output": "They remembered your interest in {genre} books and recommended a new {author} title. This shows they pay attention to regular patrons' reading preferences and provide personalized recommendations."
            },
        ]
    },
    "pharmacist": {
        "first_meeting_topics": [
            {
                "topic": "prescription_pickup",
                "conversation": [
                    ("person", "Hello, I'm {name}, the pharmacist. Picking up a prescription?"),
                    ("patient", "Yes, under my name."),
                    ("person", "Let me go over the instructions with you."),
                    ("patient", "Okay, thank you."),
                    ("person", "Take this {frequency} with food."),
                    ("patient", "Got it."),
                ],
                "context_output": "{name} is a pharmacist who provided your prescription and carefully reviewed the medication instructions. They specified to take it {frequency} with food. This professional interaction shows their attention to proper medication education."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "medication_check",
                "conversation": [
                    ("person", "How is the new medication working for you?"),
                    ("patient", "Much better, thank you for asking."),
                    ("person", "Good! Any side effects?"),
                    ("patient", "None that I've noticed."),
                ],
                "context_output": "They followed up on how your new medication is working and asked about side effects. You reported improvement with no side effects. This shows their commitment to monitoring patient outcomes beyond just dispensing medications."
            },
        ]
    },
    "yoga_instructor": {
        "first_meeting_topics": [
            {
                "topic": "first_class",
                "conversation": [
                    ("person", "Welcome! I'm {name}, your instructor. First time here?"),
                    ("patient", "Yes, my first yoga class."),
                    ("person", "Don't worry, we'll take it at your pace."),
                    ("patient", "Thank you, I appreciate that."),
                    ("person", "Let me know if you need any modifications."),
                    ("patient", "I will."),
                ],
                "context_output": "{name} is a yoga instructor who welcomed you to your first class. They were reassuring about going at your own pace and offered to provide modifications as needed. This created a comfortable and supportive environment for beginners."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "progress_encouragement",
                "conversation": [
                    ("person", "Your balance has really improved!"),
                    ("patient", "Thank you, I've been practicing."),
                    ("person", "It shows! Ready to try a more advanced pose?"),
                    ("patient", "I think so, yes."),
                ],
                "context_output": "They noticed your improved balance and acknowledged your practice efforts. They suggested progressing to more advanced poses. This shows they monitor student progress and provide appropriate challenges as skills develop."
            },
        ]
    },
    "mechanic": {
        "first_meeting_topics": [
            {
                "topic": "car_problem",
                "conversation": [
                    ("person", "Hi, I'm {name}. What's going on with your car?"),
                    ("patient", "It's making a strange {noise_type} noise."),
                    ("person", "Let me take a look and I'll diagnose the issue."),
                    ("patient", "Thank you."),
                ],
                "context_output": "{name} is a mechanic who you brought your car to because of a strange {noise_type} noise. They offered to diagnose the problem. This professional service interaction began with identifying your vehicle issue."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "repair_complete",
                "conversation": [
                    ("person", "All fixed! It was the {car_part}."),
                    ("patient", "How much do I owe you?"),
                    ("person", "Less than I quoted, actually. {price}."),
                    ("patient", "That's great, thank you."),
                ],
                "context_output": "They completed the repair of your {car_part} and charged less than originally quoted at {price}. This honest pricing builds trust and shows integrity in their business practices."
            },
        ]
    },
    "restaurant_server": {
        "first_meeting_topics": [
            {
                "topic": "new_server",
                "conversation": [
                    ("person", "Good evening, I'm {name}, I'll be your server tonight."),
                    ("patient", "Hello."),
                    ("person", "Can I start you off with something to drink?"),
                    ("patient", "Just water, please."),
                ],
                "context_output": "{name} is a restaurant server who introduced themselves and took your drink order. This standard professional service interaction established their role as your server for the meal."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "regular_customer",
                "conversation": [
                    ("person", "Good to see you again! The usual?"),
                    ("patient", "Yes, please."),
                    ("person", "I'll put that right in for you."),
                    ("patient", "Thank you."),
                ],
                "context_output": "They recognized you as a regular customer and remembered your usual order. This shows attentiveness to repeat customers and good customer service memory."
            },
        ]
    },
    "personal_trainer": {
        "first_meeting_topics": [
            {
                "topic": "fitness_assessment",
                "conversation": [
                    ("person", "I'm {name}, your personal trainer. What are your fitness goals?"),
                    ("patient", "I want to improve my {fitness_goal}."),
                    ("person", "Great! We'll create a personalized plan for that."),
                    ("patient", "Sounds good."),
                ],
                "context_output": "{name} is your personal trainer who discussed your fitness goals. You expressed interest in improving your {fitness_goal} and they committed to creating a personalized training plan. This initial consultation established your fitness objectives."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "workout_progress",
                "conversation": [
                    ("person", "You've gotten much stronger since we started!"),
                    ("patient", "I can definitely feel the difference."),
                    ("person", "Ready to increase the weights?"),
                    ("patient", "Let's do it."),
                ],
                "context_output": "They observed significant strength improvements since you began training together. They suggested increasing workout intensity by adding more weight. This shows progressive training and positive fitness outcomes."
            },
        ]
    },
    "taxi_driver": {
        "first_meeting_topics": [
            {
                "topic": "ride_request",
                "conversation": [
                    ("person", "Hi, I'm {name}, your driver. Where are we headed?"),
                    ("patient", "To {destination}, please."),
                    ("person", "I'll take {route}."),
                    ("patient", "That works for me."),
                ],
                "context_output": "{name} is a taxi driver who picked you up for a ride to {destination}. They explained the route they planned to take. This professional transport service began with confirming your destination."
            },
        ],
        "follow_up_topics": [
            {
                "topic": "regular_rider",
                "conversation": [
                    ("person", "Haven't seen you in a while! How have you been?"),
                    ("patient", "Good, thanks for asking."),
                    ("person", "Same place as usual?"),
                    ("patient", "Yes, please."),
                ],
                "context_output": "They recognized you as a regular passenger and remembered your typical destination. They also showed personal interest by asking how you've been. This indicates a friendly driver-passenger relationship developed over multiple rides."
            },
        ]
    },
}

# Variables for string formatting
VARIABLES = {
    "project": ["Johnson", "Anderson", "Phoenix", "Titan", "Mercury", "Atlas", "Summit", "Cascade", "Horizon", "Vision 2030"],
    "health_issue": ["blood pressure", "cholesterol", "joint pain", "blood sugar", "heart rhythm", "arthritis", "sleep quality"],
    "hobby": ["painting", "pottery", "piano", "yoga", "photography", "dancing", "cooking", "gardening", "knitting"],
    "food": ["cookies", "apple pie", "banana bread", "chocolate cake", "muffins", "brownies"],
    "food_item": ["bread", "cheese", "coffee", "tea", "jam", "olive oil", "honey"],
    "child_name": ["Emma", "Liam", "Olivia", "Noah", "Sophia", "Mason", "Ava", "Lucas", "Isabella"],
    "day": ["Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "Monday"],
    "time": ["morning", "afternoon", "evening"],
    "location": ["mountains", "national park", "state forest", "lakeside", "countryside"],
    "car_part": ["transmission", "brakes", "alternator", "battery", "radiator", "spark plugs", "timing belt"],
    "dish": ["lasagna", "pot roast", "chicken soup", "beef stew", "pasta sauce", "meat loaf"],
    "genre": ["mystery", "historical fiction", "science fiction", "thriller", "biography", "romance"],
    "activity": ["play chess", "debate philosophy", "study together", "play basketball", "go hiking"],
    "number": ["3B", "12A", "5C", "8D", "2A", "10B"],
    "care_task": ["medication management", "mobility assistance", "meal preparation", "personal care"],
    "company": ["IBM", "Microsoft", "General Electric", "Boeing", "Ford", "3M"],
    "department": ["produce", "bakery", "deli", "meat", "dairy"],
    "produce_item": ["strawberries", "tomatoes", "corn", "peaches", "blueberries"],
    "test_type": ["blood work", "cholesterol screening", "blood pressure monitoring", "glucose test"],
    "body_part": ["knee", "shoulder", "back", "hip", "ankle"],
    "duration": ["6-8", "8-12", "4-6"],
    "measurement": ["range of motion", "strength", "flexibility", "balance"],
    "author": ["Agatha Christie novel", "John Grisham thriller", "Stephen King book"],
    "frequency": ["twice daily", "once daily", "three times daily", "every morning"],
    "noise_type": ["grinding", "squeaking", "rattling", "clicking"],
    "price": ["$150", "$200", "$175", "$225"],
    "destination": ["the medical center", "the grocery store", "downtown", "the library"],
    "route": ["Main Street", "the highway", "the scenic route"],
    "fitness_goal": ["strength", "endurance", "flexibility", "balance", "overall fitness"],
    "name": ["Jennifer", "David", "Maria", "James", "Linda", "Robert", "Patricia", "Michael", "Sarah", "William",
             "Lisa", "Richard", "Nancy", "Thomas", "Karen", "Charles", "Betty", "Christopher", "Margaret", "Daniel"]
}

def fill_template(text: str, used_values: dict = None) -> tuple:
    """Fill template with random values from VARIABLES."""
    if used_values is None:
        used_values = {}

    result = text
    for key in VARIABLES:
        if "{" + key + "}" in result:
            if key not in used_values:
                used_values[key] = random.choice(VARIABLES[key])
            result = result.replace("{" + key + "}", used_values[key])

    return result, used_values


def format_conversation(conversation: List[tuple]) -> str:
    """Format conversation tuples into readable text."""
    lines = []
    for speaker, text in conversation:
        lines.append(f"- {speaker}: {text}")
    return "\n".join(lines)


def generate_first_meeting_example(relationship: str) -> dict:
    """Generate a first meeting training example."""
    templates = CONVERSATION_TEMPLATES[relationship]["first_meeting_topics"]
    template = random.choice(templates)

    # Fill in variables
    used_values = {}
    conversation_lines = []
    for speaker, text in template["conversation"]:
        filled_text, used_values = fill_template(text, used_values)
        conversation_lines.append((speaker, filled_text))

    context_output, _ = fill_template(template["context_output"], used_values)

    # Format conversation
    conversation_text = format_conversation(conversation_lines)

    system_prompt = (
        "You are a context aggregator for a memory system. "
        "Given previous context about a person and their current conversation, "
        "generate a detailed, verbose summary (3-5 sentences) that captures "
        "the conversation history, relationship details, and important information."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Previous Context: This is the first time you're meeting this person.\n\n"
                f"Current Conversation:\n{conversation_text}"},
            {"role": "assistant", "content": context_output}
        ]
    }


def generate_follow_up_example(relationship: str, previous_context: str) -> dict:
    """Generate a follow-up meeting training example."""
    templates = CONVERSATION_TEMPLATES[relationship]["follow_up_topics"]
    template = random.choice(templates)

    # Fill in variables
    used_values = {}
    conversation_lines = []
    for speaker, text in template["conversation"]:
        filled_text, used_values = fill_template(text, used_values)
        conversation_lines.append((speaker, filled_text))

    context_addition, _ = fill_template(template["context_output"], used_values)

    # Combine previous context with new information
    updated_context = f"{previous_context} {context_addition}"

    # Format conversation
    conversation_text = format_conversation(conversation_lines)

    system_prompt = (
        "You are a context aggregator for a memory system. "
        "Given previous context about a person and their current conversation, "
        "generate a detailed, verbose summary (3-5 sentences) that captures "
        "the conversation history, relationship details, and important information."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content":
                f"Previous Context: {previous_context}\n\n"
                f"Current Conversation:\n{conversation_text}"},
            {"role": "assistant", "content": updated_context}
        ]
    }


def generate_training_dataset(num_examples: int = 1000) -> List[dict]:
    """Generate a full training dataset."""
    training_data = []

    # Get all relationship types
    relationships = list(CONVERSATION_TEMPLATES.keys())
    examples_per_relationship = num_examples // len(relationships)

    for relationship in relationships:
        # First meeting examples (40% of examples for this relationship)
        num_first_meetings = int(examples_per_relationship * 0.4)
        for _ in range(num_first_meetings):
            example = generate_first_meeting_example(relationship)
            training_data.append(example)

        # Follow-up examples (60% of examples for this relationship)
        # Only if follow-up topics exist for this relationship
        if CONVERSATION_TEMPLATES[relationship]["follow_up_topics"]:
            num_follow_ups = examples_per_relationship - num_first_meetings
            for _ in range(num_follow_ups):
                # Generate a chain of 1-3 previous meetings
                chain_length = random.randint(1, 3)

                # Start with first meeting
                first_example = generate_first_meeting_example(relationship)
                current_context = first_example["messages"][2]["content"]  # assistant's response

                # Build up context through chain
                for _ in range(chain_length - 1):
                    temp_example = generate_follow_up_example(relationship, current_context)
                    current_context = temp_example["messages"][2]["content"]

                # Final follow-up is what we add to training
                example = generate_follow_up_example(relationship, current_context)
                training_data.append(example)
        else:
            # If no follow-ups, generate more first meetings
            for _ in range(examples_per_relationship - num_first_meetings):
                example = generate_first_meeting_example(relationship)
                training_data.append(example)

    # Shuffle the data
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

    # Generate 1000 examples
    print("Generating training dataset with diverse relationships...")
    print(f"Relationship types: {len(CONVERSATION_TEMPLATES)}")
    training_examples = generate_training_dataset(1000)

    # Save to JSONL
    output_path = "data/context_aggregation_training.jsonl"
    save_to_jsonl(training_examples, output_path)

    print("\n" + "="*80)
    print("SAMPLE TRAINING EXAMPLES:")
    print("="*80)

    # Show 3 sample examples
    for i in range(min(3, len(training_examples))):
        print(f"\nExample {i+1}:")
        print(json.dumps(training_examples[i], indent=2))
        print("-"*80)
