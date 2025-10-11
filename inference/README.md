# Real-Time Inference Service Prototype

A lightweight prototype demonstrating real-time conversation data processing using Server-Sent Events (SSE).

## Architecture

```
Mock Metadata Service → [SSE] → Inference Service → [SSE] → Frontend/Consumer
   (Port 8000)                     (Port 8002)
```

## Components

1. **Mock Metadata Service** (`mock_metadata_service.py`)
   - Simulates speaker diarization metadata service
   - Emits conversation events via SSE at `/stream/conversation`
   - Includes: person_id, text, timestamp, confidence

2. **Inference Service** (`main.py`)
   - Consumes conversation events from metadata service
   - Processes data with hardcoded inference logic
   - Streams results via SSE at `/stream/inference`

3. **Mock Consumer** (`mock_consumer.py`)
   - Demonstrates how to consume inference results
   - Can represent frontend or downstream service

## Setup

### Option 1: Automated Setup (Recommended)
```bash
cd inference
./setup.sh
```

This will:
- Create a virtual environment in `venv/`
- Install all dependencies
- Provide instructions for running

### Option 2: Manual Setup
```bash
cd inference

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Prototype

**Important**: Make sure to activate the virtual environment first:
```bash
source venv/bin/activate
```

You need to run the services in separate terminals (all with venv activated):

### Terminal 1: Start Mock Metadata Service
```bash
source venv/bin/activate
python mock_metadata_service.py
```
This starts on http://localhost:8000

### Terminal 2: Start Inference Service
```bash
source venv/bin/activate
python main.py
```
This starts on http://localhost:8002

### Terminal 3: Start Mock Consumer (Optional)
```bash
source venv/bin/activate
python mock_consumer.py
```
This connects to the inference service and displays results in real-time.

## API Endpoints

### Mock Metadata Service (Port 8000)
- `GET /stream/conversation` - SSE stream of conversation events
- `GET /health` - Health check

### Inference Service (Port 8002)
- `GET /stream/inference` - SSE stream of processed inference results
- `GET /health` - Health check with queue status
- `GET /` - Service information

## Data Models

### ConversationEvent (Input)
```json
{
  "event_id": "evt_abc123",
  "person_id": "person_001",
  "text": "Hello, how can I help you today?",
  "timestamp": "2025-10-11T14:30:00Z",
  "confidence": 0.95
}
```

### InferenceResult (Output)
```json
{
  "result_id": "res_xyz789",
  "event_id": "evt_abc123",
  "person_id": "person_001",
  "original_text": "Hello, how can I help you today?",
  "analysis": "Customer inquiry detected - requires assistance",
  "sentiment": "positive",
  "keywords": ["hello", "help", "today"],
  "timestamp": "2025-10-11T14:30:01Z"
}
```

## Testing with cURL

### Subscribe to conversation events:
```bash
curl -N http://localhost:8000/stream/conversation
```

### Subscribe to inference results:
```bash
curl -N http://localhost:8002/stream/inference
```

## Customizing the Inference Logic

The inference logic is hardcoded in `main.py` in the `hardcoded_inference_logic()` function.

To add your own logic:
1. Open `main.py`
2. Find the `hardcoded_inference_logic()` function
3. Replace or enhance the logic with your own processing
4. Update the `InferenceResult` model if needed

Example areas to customize:
- Sentiment analysis rules
- Keyword extraction
- Intent detection
- Entity recognition
- Custom business logic

## Notes

- The mock metadata service generates events every 2-5 seconds
- All timestamps are in UTC
- The inference service uses an asyncio queue to handle streaming
- SSE connections are maintained with keepalive comments every 30 seconds
- Services will automatically retry connections if they fail
