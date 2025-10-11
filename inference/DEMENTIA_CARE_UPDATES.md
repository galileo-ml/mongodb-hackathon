# Dementia Care Updates

## Overview
The inference service has been updated to support a dementia care use case with hybrid analysis that combines past relationship context with real-time conversation tracking.

## Key Changes

### 1. Data Models (`models.py`)

**ConversationEvent** - Added:
- `conversation_id`: Groups utterances into conversation sessions

**InferenceResult** - Completely redesigned with three sections:

#### Past Context (Patient Reminder)
- `relationship_type`: Who the person is (daughter, son, friend)
- `relationship_notes`: Human-readable relationship description
- `last_conversation_date`: When they last spoke
- `last_conversation_summary`: What they discussed before
- `recurring_topics`: Topics commonly discussed with this person

#### Current Utterance
- `current_utterance`: What was just said
- `utterance_sentiment`: Sentiment of this specific utterance

#### Running Conversation Analysis
- `current_conversation_summary`: Summary of today's conversation so far
- `topics_discussed_today`: Topics covered in current conversation
- `key_moments_today`: Important moments/statements
- `emotional_tone_today`: Overall emotional tone
- `interaction_notes_today`: Observations about patient engagement

### 2. Inference Service (`main.py`)

**New Features:**
- **Conversation State Tracking**: Accumulates utterances per `conversation_id`
- **Mock Person History**: Hardcoded database of past interactions (3 people)
  - person_001: Sarah (daughter)
  - person_002: Michael (son)
  - person_003: Robert (friend)
- **Hybrid Analysis Functions**:
  - `analyze_utterance_sentiment()`: Per-utterance sentiment
  - `build_conversation_summary()`: Running conversation summary
  - `extract_topics()`: Topic detection (family, work, health, hobbies, plans)
  - `extract_key_moments()`: Important statement detection
  - `assess_emotional_tone()`: Overall conversation tone
  - `generate_interaction_notes()`: Patient engagement assessment

### 3. Mock Metadata Service (`mock_metadata_service.py`)

**Updates:**
- **Person-Specific Conversations**: Each person has contextually appropriate utterances
- **Conversation Sessions**: Tracks active conversations per person
- **Realistic Dialogue**: Mock data reflects dementia care scenario

### 4. Mock Consumer (`mock_consumer.py`)

**Enhanced Display:**
- Structured output showing all three analysis sections
- Visual indicators (📍👤💬📊🩺) for different information types
- Clear formatting for patient reminders and conversation tracking

## Use Case Flow

### When Patient Encounters Someone:

1. **Recognition** (via diarization metadata)
   - System identifies `person_id`

2. **Past Context Retrieval**
   - Display: "This is Sarah, your daughter who visits every Tuesday"
   - Show: Last conversation summary and common topics

3. **Live Conversation Processing**
   - Each utterance analyzed in real-time
   - Running summary builds throughout conversation
   - Patient engagement tracked

4. **Output to Frontend**
   - SSE stream provides continuous updates
   - Frontend can display reminders + live analysis

## Running the Updated System

```bash
# Terminal 1: Mock Metadata Service (Port 8000)
source venv/bin/activate
python mock_metadata_service.py

# Terminal 2: Inference Service (Port 8002)
source venv/bin/activate
python main.py

# Terminal 3: Mock Consumer (displays results)
source venv/bin/activate
python mock_consumer.py
```

## Example Output

```
INFERENCE RESULT [res_abc123]

📍 TRACKING INFO:
   Person ID:        person_001
   Conversation ID:  conv_xyz789

👤 PAST CONTEXT (For Patient Reminder):
   Relationship:     DAUGHTER
   Who:              Sarah, your daughter who visits every Tuesday afternoon
   Last Spoke:       2025-10-08 15:00:00
   Last Conversation: You discussed Sarah's promotion and grandchildren visit
   Common Topics:    grandchildren, work, family gatherings, gardening

💬 CURRENT UTTERANCE:
   Text:             "The kids are so excited to visit you next weekend!"
   Sentiment:        POSITIVE

📊 TODAY'S CONVERSATION ANALYSIS:
   Summary:          Sarah is visiting. You've exchanged 5 messages...
   Emotional Tone:   WARM AND POSITIVE
   Topics Today:     family, plans, work
   Key Moments:
      1. I got that promotion at work I mentioned!
      2. The kids are so excited to visit you next weekend!

🩺 PATIENT INTERACTION NOTES:
   Patient is engaged and responding appropriately
```

## Next Steps for Production

1. **Replace hardcoded logic** with actual ML models:
   - LLM for conversation summarization
   - NER for topic extraction
   - Advanced sentiment analysis

2. **Add MongoDB storage**:
   - Store conversation summaries
   - Maintain person history database
   - Query past interactions

3. **Enhanced analysis**:
   - Memory recall assessment
   - Confusion detection
   - Emotional state tracking

4. **Frontend integration**:
   - Real-time display components
   - Caregiver dashboard
   - Alert system for concerning patterns
