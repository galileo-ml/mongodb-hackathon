# Optional IDs Update

## Overview
The `event_id` and `conversation_id` fields in `ConversationEvent` are now **optional**. The metadata service only needs to provide the essential diarization data.

## Changes Made

### 1. ConversationEvent Model
**Required fields:**
- `person_id` - Who is speaking
- `text` - What they said
- `timestamp` - When they said it
- `confidence` - Diarization confidence score

**Optional fields:**
- `event_id` - Generated automatically if not provided
- `conversation_id` - Generated automatically if not provided

### 2. Inference Service Behavior
When `event_id` or `conversation_id` are missing, the inference service will:

- **event_id**: Generate a unique ID using `evt_{random}`
- **conversation_id**: Use `conv_{person_id}` to group all utterances from the same person

This means each person gets their own ongoing conversation session automatically.

### 3. Mock Metadata Service
Now sends minimal data:
```json
{
  "person_id": "person_001",
  "text": "Hi dad, how are you feeling today?",
  "timestamp": "2025-10-11T14:30:00Z",
  "confidence": 0.95
}
```

The inference service handles ID generation internally.

## Benefits

1. **Simpler Integration**: External metadata services don't need to manage IDs
2. **Flexibility**: Services can provide IDs if they want, or let inference service generate them
3. **Automatic Conversation Grouping**: Same person = same conversation by default

## Example Usage

### Minimal Event (IDs auto-generated):
```python
event = ConversationEvent(
    person_id="person_sarah",
    text="The kids are excited to visit!",
    timestamp=datetime.utcnow(),
    confidence=0.92
)
# inference service creates:
# - event_id: "evt_abc12345"
# - conversation_id: "conv_person_sarah"
```

### Full Event (IDs provided):
```python
event = ConversationEvent(
    person_id="person_sarah",
    text="The kids are excited to visit!",
    timestamp=datetime.utcnow(),
    confidence=0.92,
    event_id="my_custom_event_001",
    conversation_id="session_20251011_sarah"
)
# inference service uses provided IDs
```

## Migration Notes

If you have existing code that always provides these IDs, it will continue to work. The change is backward compatible - optional fields just means they're not required, not that they're ignored if provided.
