# AR Glasses - Simplified Design

## Overview
The inference service has been simplified to provide minimal, focused information perfect for AR glasses display.

## Data Models

### ConversationEvent (Input)
**Required:**
- `person_id` - Who is speaking
- `text` - What they said
- `timestamp` - When they said it
- `confidence` - Diarization confidence

**Optional (auto-generated if not provided):**
- `event_id`
- `conversation_id`

### InferenceResult (Output)
**Only 4 fields:**
- `person_id` - Internal tracking
- `name` - Person's name (e.g., "Sarah")
- `relationship` - Relationship to patient (e.g., "Your daughter")
- `description` - One-line context (e.g., "Last spoke 3 days ago about her promotion")

## Example Flow

### 1. Metadata Service Sends Event
```json
{
  "person_id": "person_001",
  "text": "Hi dad, how are you feeling?",
  "timestamp": "2025-10-11T14:30:00Z",
  "confidence": 0.95
}
```

### 2. Inference Service Processes
- Looks up person_001 in mock database
- Tracks conversation state
- Generates simple result

### 3. AR Glasses Display
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

## Mock Person Database

The prototype includes 3 people:

**person_001 - Sarah**
- Relationship: Your daughter
- Last interaction: About promotion and grandchildren visiting

**person_002 - Michael**
- Relationship: Your son
- Last interaction: Brought groceries, talked about camping trip

**person_003 - Robert**
- Relationship: Your friend from book club
- Last interaction: Discussed mystery novel and college memories

## Description Updates

The `description` field updates dynamically:

**Before conversation:**
- Shows last interaction from database

**During conversation:**
- "Just started talking" (1 message)
- "Having a conversation (3 messages exchanged)" (2-4 messages)
- "Deep in conversation (8 messages)" (5+ messages)

## Running the System

```bash
# Terminal 1: Metadata Service (sends diarization events)
source venv/bin/activate
python mock_metadata_service.py

# Terminal 2: Inference Service (processes and enriches)
source venv/bin/activate
python main.py

# Terminal 3: AR Glasses Simulator (displays results)
source venv/bin/activate
python mock_consumer.py
```

## Benefits

1. **Minimal Display** - Only essential information
2. **No Cognitive Overload** - Simple, clean output
3. **Fast Processing** - Lightweight logic
4. **Easy to Extend** - Add more people to MOCK_PERSON_DATA
5. **AR-Optimized** - Perfect for limited screen space

## Next Steps for Production

1. **Real Database**: Replace `MOCK_PERSON_DATA` with MongoDB queries
2. **LLM Integration**: Use LLM to generate more contextual descriptions
3. **Face/Voice Matching**: Connect person_id to actual face recognition
4. **Dynamic Updates**: Update person data after each conversation
5. **Privacy Controls**: Add consent management and data retention policies
