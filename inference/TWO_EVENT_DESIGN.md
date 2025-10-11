# Two Event Types Design

## Overview
The inference service now handles two distinct event types for AR glasses dementia care.

## Event Types

### 1. PERSON_DETECTED
**When**: Someone enters the camera frame
**Purpose**: Display AR information immediately
**Data**:
```json
{
  "event_type": "PERSON_DETECTED",
  "person_id": "person_001",
  "timestamp": "2025-10-11T14:30:00Z"  // optional
}
```

**Action**: Inference service returns `InferenceResult` with name, relationship, last interaction

**AR Display**:
```
==========================================
AR GLASSES DISPLAY
==========================================

👤 NAME:         Sarah
💙 RELATIONSHIP: Your daughter
📝 CONTEXT:      Last spoke 3 days ago about her promotion

   [Person ID: person_001]
==========================================
```

### 2. CONVERSATION_END
**When**: Conversation finishes
**Purpose**: Store full conversation for future reference
**Data**:
```json
{
  "event_type": "CONVERSATION_END",
  "person_id": "person_001",
  "text": "Sarah: Hi dad...\nPatient: I'm doing well...",
  "timestamp": "2025-10-11T14:35:00Z"  // optional
}
```

**Action**: Inference service stores conversation, updates person's last_interaction
**No AR Display**: This event is for storage only

## Data Flow

```
1. Person enters frame
   → PERSON_DETECTED event sent
   → Inference service looks up person
   → AR displays: name, relationship, last interaction

2. Conversation happens
   (Camera records, diarization processes)

3. Person leaves / conversation ends
   → CONVERSATION_END event sent with full transcript
   → Inference service stores conversation
   → Updates "last interaction" for next time
```

## Running the System

```bash
# Terminal 1: Mock Metadata Service (Port 8000)
source venv/bin/activate
python mock_metadata_service.py

# Terminal 2: Inference Service (Port 8002)
source venv/bin/activate
python main.py

# Terminal 3: AR Glasses Simulator
source venv/bin/activate
python mock_consumer.py
```

## Mock Behavior

The mock metadata service simulates realistic timing:
1. Emit PERSON_DETECTED for a random person
2. Wait 8-15 seconds (simulating conversation)
3. Emit CONVERSATION_END with full transcript
4. Wait 5-10 seconds (pause between visitors)
5. Repeat

## Key Features

**Simplicity**:
- No line-by-line utterance tracking
- No complex conversation state
- Just two events: detect and store

**Flexibility**:
- `timestamp` is optional (auto-generated if missing)
- `text` format is free-form (works with any transcript format)

**Efficiency**:
- Only PERSON_DETECTED events generate AR output
- CONVERSATION_END events update database silently
- Minimal processing overhead

## Next Steps for Production

1. **Real Diarization Integration**: Connect to actual camera + diarization service
2. **Database Storage**: Replace `MOCK_PERSON_DATA` with MongoDB
3. **LLM Summarization**: Use LLM to generate better descriptions from conversation text
4. **Face Recognition**: Link person_id to actual face embeddings
5. **Privacy Controls**: Add consent management and data retention policies

## Example Flow

```
14:30:00 - PERSON_DETECTED: person_001
           AR shows: "Sarah | Your daughter | Last spoke 3 days ago..."

14:30:05 - Conversation starts (not sent as events)
14:30:10 - Still talking...
14:30:15 - Still talking...

14:30:20 - CONVERSATION_END: person_001
           Text: "Sarah: Hi dad...\nPatient: I'm well..."
           Stored → becomes new "last interaction"

14:30:30 - Sarah leaves

14:30:40 - PERSON_DETECTED: person_002
           AR shows: "Michael | Your son | Visited yesterday..."
```

Clean, simple, effective!
