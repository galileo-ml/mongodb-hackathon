# Two Event Types - Final Design

## Overview
The system now uses **two distinct event types** for clean separation of concerns:
1. **PERSON_DETECTED** - Triggers AR display
2. **CONVERSATION_END** - Stores conversation for future reference

## Data Models

### ConversationUtterance
```python
{
  "speaker": "person_001" | "patient",
  "text": "What was said",
  "timestamp": "2025-10-11T14:30:00Z"
}
```

### ConversationEvent

**Type 1: PERSON_DETECTED**
```json
{
  "event_type": "PERSON_DETECTED",
  "person_id": "person_001",
  "timestamp": "2025-10-11T14:30:00Z"
}
```

**Type 2: CONVERSATION_END**
```json
{
  "event_type": "CONVERSATION_END",
  "person_id": "person_001",
  "timestamp": "2025-10-11T14:35:00Z",
  "conversation": [
    {
      "speaker": "person_001",
      "text": "Hi dad, how are you feeling?",
      "timestamp": "2025-10-11T14:30:05Z"
    },
    {
      "speaker": "patient",
      "text": "I'm doing well!",
      "timestamp": "2025-10-11T14:30:10Z"
    }
  ]
}
```

### InferenceResult (unchanged)
```json
{
  "person_id": "person_001",
  "name": "Sarah",
  "relationship": "Your daughter",
  "description": "Last spoke 3 days ago about her promotion"
}
```

## Flow

### 1. Person Enters Frame
```
Metadata Service → PERSON_DETECTED event
                     ↓
                Inference Service → Looks up person in database
                     ↓
                InferenceResult → Stream to AR glasses
                     ↓
                AR Display Shows: Name, Relationship, Last Interaction
```

### 2. Conversation Happens
```
(Patient and person talk for a while...)
```

### 3. Conversation Ends
```
Metadata Service → CONVERSATION_END event (with full transcript)
                     ↓
                Inference Service → Generate summary
                     ↓
                Update person database with new "last interaction"
                     ↓
                (No AR display - storage only)
```

### 4. Next Time Person Enters Frame
```
PERSON_DETECTED → Shows updated "last interaction" from previous conversation
```

## Inference Service Handlers

### handle_person_detected()
- Look up person in MOCK_PERSON_DATA
- Return InferenceResult with name, relationship, last_interaction
- Stream to AR glasses

### handle_conversation_end()
- Extract conversation array
- Generate summary (keyword-based for prototype)
- Update MOCK_PERSON_DATA["last_interaction"]
- Log storage (in production: save to MongoDB)

## Mock Metadata Service Flow

1. Pick random person
2. Send PERSON_DETECTED event
3. Wait 15-25 seconds (simulating conversation)
4. Generate mock conversation (3-5 exchanges)
5. Send CONVERSATION_END event with full conversation
6. Wait 10-20 seconds before next person

## Running the System

```bash
# Terminal 1: Metadata Service (Port 8000)
source venv/bin/activate
python mock_metadata_service.py

# Terminal 2: Inference Service (Port 8002)
source venv/bin/activate
python main.py

# Terminal 3: AR Glasses Simulator
source venv/bin/activate
python mock_consumer.py
```

## Example Output

**When Sarah enters frame (PERSON_DETECTED):**
```
================================================================================
AR GLASSES DISPLAY
================================================================================

👤 NAME:         Sarah
💙 RELATIONSHIP: Your daughter
📝 CONTEXT:      Last spoke 3 days ago about her promotion and the grandchildren visiting

   [Person ID: person_001]
================================================================================
```

**After conversation ends (CONVERSATION_END):**
- No AR display
- Inference service logs: "Updated last interaction for Sarah: Just talked about work and family (8 messages)"
- Next time Sarah enters, description shows new "last interaction"

## Key Benefits

1. **Clean Separation**: Detection vs. Storage are separate events
2. **No Line-by-Line Processing**: Simpler, matches real data flow
3. **Structured Conversations**: JSON array format for easy parsing
4. **Automatic History Updates**: Each conversation becomes next "last interaction"
5. **AR-Optimized**: Only relevant info streamed to glasses

## Next Steps for Production

1. **MongoDB Integration**: Replace MOCK_PERSON_DATA with real database
2. **LLM Summarization**: Use LLM to generate better conversation summaries
3. **Face/Voice Matching**: Connect person_id to actual recognition service
4. **Conversation Storage**: Store full conversations in MongoDB for analysis
5. **Privacy & Consent**: Add data retention and deletion policies
