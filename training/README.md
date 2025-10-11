# Fine-tuning Context Aggregation Model on Fireworks.ai

This directory contains scripts and datasets for fine-tuning Model 1 (Context Aggregation Model) on Fireworks.ai.

## Model Purpose

**Context Aggregation Model** takes:
- **Input**: Previous Context + Current Conversation
- **Output**: Updated detailed context (verbose, 3-5 sentences) for MongoDB storage

This model will be called during `CONVERSATION_END` events to update the `aggregated_context` field in MongoDB.

## Setup

### 1. Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

### 2. Configure API Key

Copy the example environment file and add your Fireworks API key:

```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
FIREWORKS_API_KEY=your_actual_api_key_here
```

Get your API key from: https://fireworks.ai/account/api-keys

### 3. Review Configuration (Optional)

The default configuration in `.env` is:
- **Base Model**: Llama 3.1 8B Instruct
- **LoRA Rank**: 8
- **Learning Rate**: 0.0001
- **Epochs**: 3
- **Batch Size**: 4

You can adjust these values in `.env` if needed.

## Usage

### Generate Training Data (Already Done)

The training data has been generated with 990 examples covering 18 different relationship types:

```bash
python generate_context_aggregation_data.py
```

Output: `data/context_aggregation_training.jsonl`

### Upload Dataset and Start Fine-tuning

Run the fine-tuning script:

```bash
python fireworks_finetune.py
```

This script will:
1. ✅ Validate the JSONL format
2. ✅ Upload the dataset to Fireworks.ai
3. ✅ Create a fine-tuning job
4. ✅ Monitor the job until completion

The script will output:
- Dataset ID
- Job ID
- Training progress
- **Fine-tuned Model ID** (when complete)

### Monitor Existing Job

If you interrupted the monitoring, you can check job status with:

```python
from fireworks.client import Fireworks
from config import FIREWORKS_API_KEY

client = Fireworks(api_key=FIREWORKS_API_KEY)
job = client.fine_tuning.jobs.retrieve("your_job_id")
print(f"Status: {job.status}")
print(f"Model: {job.fine_tuned_model}")
```

## Training Data Structure

Each training example follows this format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a context aggregator for a memory system..."
    },
    {
      "role": "user",
      "content": "Previous Context: ...\n\nCurrent Conversation:\n- person: ...\n- patient: ..."
    },
    {
      "role": "assistant",
      "content": "Detailed verbose context summary (3-5 sentences)"
    }
  ]
}
```

## Relationship Types in Dataset

The dataset includes diverse relationships:

**Family**: daughter, son

**Friends**: friend

**Service Workers**: postal_worker, grocery_clerk, restaurant_server, taxi_driver

**Healthcare**: doctor, physical_therapist, pharmacist, caregiver

**Professional Services**: hairdresser, librarian, mechanic, yoga_instructor, personal_trainer

**Community**: neighbor, ex_colleague

## After Fine-tuning

Once the fine-tuning job completes, you'll receive a **fine-tuned model ID** like:

```
accounts/your-account/models/context-aggregation-model-v1-abc123
```

Save this model ID! You'll need it to integrate the model into the inference service.

## Next Steps

After successful fine-tuning:

1. **Save the Model ID**: Copy the fine-tuned model ID from the output
2. **Test the Model**: Try some inference calls to validate the model works
3. **Integrate into Inference Service**: Update `inference/main.py` to use the fine-tuned model
4. **Fine-tune Model 2**: Create training data for the AR Display Model

## Troubleshooting

**Error: FIREWORKS_API_KEY not found**
- Make sure you created a `.env` file (not `.env.example`)
- Check that your API key is valid

**Error: Dataset validation failed**
- Check the JSONL format in `data/context_aggregation_training.jsonl`
- Each line must be valid JSON with the correct message structure

**Fine-tuning job failed**
- Check the Fireworks.ai dashboard for error details
- Verify your account has sufficient credits
- Check that the base model name is correct

## Cost Estimation

Fine-tuning costs on Fireworks.ai:
- Pay per training token
- ~990 examples × ~200 tokens/example × 3 epochs = ~600K training tokens
- Check current pricing at: https://fireworks.ai/pricing

## Files

- `generate_context_aggregation_data.py` - Generate synthetic training data
- `fireworks_finetune.py` - Upload dataset and launch fine-tuning
- `config.py` - Configuration management
- `requirements.txt` - Python dependencies
- `.env.example` - Environment variable template
- `data/context_aggregation_training.jsonl` - Training dataset (990 examples)
