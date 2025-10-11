# Upload Dataset via Fireworks.ai UI

Since the REST API for dataset upload requires specific formatting, the easiest way to upload your training data is through the Fireworks.ai web interface.

## Steps

### 1. Go to Fireworks.ai Dashboard

Visit: https://fireworks.ai/datasets

### 2. Click "Create Dataset" or "Upload Dataset"

### 3. Upload Your File

- Click "Choose File" or drag and drop
- Select: `/Users/vatsalbajaj/work/mongodb-hackathon/training/data/context_aggregation_training.jsonl`
- Give it a name: `context-aggregation-model-v1-dataset`

### 4. Wait for Validation

Fireworks will validate your JSONL format (should pass - we already validated it!)

### 5. Get the Dataset Name

Once uploaded, note the dataset name (should be `context-aggregation-model-v1-dataset`)

### 6. Run the Fine-tuning Script

Once the dataset is uploaded, come back here and we'll create the fine-tuning job programmatically!

---

**Alternative: Use firectl CLI**

If you prefer command-line:

```bash
# Install firectl
pip install firectl

# Upload dataset
firectl create dataset context-aggregation-model-v1-dataset \
    --file data/context_aggregation_training.jsonl \
    --wait
```

Then run `python fireworks_finetune_from_dataset.py` with the dataset name.
