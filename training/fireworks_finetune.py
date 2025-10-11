"""Script to upload training data and launch fine-tuning job on Fireworks.ai."""

import time
import json
import requests
from pathlib import Path
from config import (
    FIREWORKS_API_KEY,
    BASE_MODEL,
    LORA_RANK,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    TRAINING_DATA_PATH,
    FINETUNE_JOB_NAME,
)

# Fireworks API configuration
BASE_URL = "https://api.fireworks.ai/v1"
HEADERS = {
    "Authorization": f"Bearer {FIREWORKS_API_KEY}",
    "Content-Type": "application/json"
}

def validate_jsonl_format(file_path: str) -> bool:
    """Validate that the JSONL file has the correct format."""
    print(f"Validating JSONL format for {file_path}...")

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            print("❌ Error: File is empty")
            return False

        # Check first few examples
        for i, line in enumerate(lines[:3]):
            try:
                data = json.loads(line)

                # Validate structure
                if "messages" not in data:
                    print(f"❌ Error: Line {i+1} missing 'messages' field")
                    return False

                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    print(f"❌ Error: Line {i+1} 'messages' must be a list with at least 2 messages")
                    return False

                # Check message structure
                for msg in messages:
                    if "role" not in msg or "content" not in msg:
                        print(f"❌ Error: Line {i+1} message missing 'role' or 'content'")
                        return False
                    if msg["role"] not in ["system", "user", "assistant"]:
                        print(f"❌ Error: Line {i+1} invalid role: {msg['role']}")
                        return False

            except json.JSONDecodeError as e:
                print(f"❌ Error: Line {i+1} is not valid JSON: {e}")
                return False

        print(f"✅ Validation passed! {len(lines)} examples found")
        return True

    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return False
    except Exception as e:
        print(f"❌ Error during validation: {e}")
        return False


def upload_dataset(file_path: str, dataset_name: str) -> str:
    """Upload training dataset to Fireworks.ai using REST API.

    Returns:
        dataset_name: The name of the uploaded dataset
    """
    print(f"\n{'='*80}")
    print("STEP 1: UPLOADING DATASET")
    print(f"{'='*80}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Training data file not found: {file_path}")

    # Validate format before upload
    if not validate_jsonl_format(file_path):
        raise ValueError("JSONL validation failed. Please fix the format and try again.")

    print(f"Uploading {file_path} to Fireworks.ai...")

    try:
        # Read the file content
        with open(file_path, 'rb') as f:
            files = {'file': (dataset_name, f, 'application/jsonl')}

            # Upload dataset using multipart/form-data
            upload_headers = {
                "Authorization": f"Bearer {FIREWORKS_API_KEY}"
            }

            response = requests.post(
                f"{BASE_URL}/datasets",
                headers=upload_headers,
                files=files,
                data={'name': dataset_name}
            )

            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                print(f"✅ Dataset uploaded successfully!")
                print(f"   Dataset Name: {dataset_name}")
                return dataset_name
            else:
                error_msg = f"Upload failed with status {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise


def get_account_id() -> str:
    """Get the account ID from Fireworks API."""
    try:
        response = requests.get(
            f"{BASE_URL}/accounts",
            headers=HEADERS
        )

        if response.status_code == 200:
            accounts = response.json()

            # Handle different response formats
            if isinstance(accounts, dict) and 'accounts' in accounts:
                account_list = accounts['accounts']
                if len(account_list) > 0:
                    # Extract account ID from 'name' field (format: accounts/account-id)
                    account_name = account_list[0].get('name', '')
                    if account_name.startswith('accounts/'):
                        account_id = account_name.replace('accounts/', '')
                        return account_id
                    elif 'id' in account_list[0]:
                        return account_list[0]['id']

            raise Exception(f"Could not extract account ID from response: {accounts}")
        else:
            raise Exception(f"Failed to get accounts: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Error getting account ID: {e}")
        raise


def create_finetune_job(dataset_name: str, account_id: str) -> str:
    """Create a fine-tuning job on Fireworks.ai using REST API.

    Returns:
        job_id: The ID of the fine-tuning job
    """
    print(f"\n{'='*80}")
    print("STEP 2: CREATING FINE-TUNING JOB")
    print(f"{'='*80}")

    print(f"Configuration:")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  LoRA Rank: {LORA_RANK}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Job Name: {FINETUNE_JOB_NAME}")

    try:
        # Create the fine-tuning job payload
        payload = {
            "dataset": dataset_name,
            "baseModel": BASE_MODEL,
            "displayName": FINETUNE_JOB_NAME,
            "epochs": EPOCHS,
            "learningRate": LEARNING_RATE,
            "loraRank": LORA_RANK,
            "batchSize": BATCH_SIZE
        }

        response = requests.post(
            f"{BASE_URL}/accounts/{account_id}/supervisedFineTuningJobs",
            headers=HEADERS,
            json=payload
        )

        if response.status_code == 200 or response.status_code == 201:
            result = response.json()
            job_id = result.get('id', result.get('jobId', 'unknown'))
            print(f"✅ Fine-tuning job created successfully!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {result.get('status', 'unknown')}")
            return job_id
        else:
            error_msg = f"Job creation failed with status {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

    except Exception as e:
        print(f"❌ Error creating fine-tuning job: {e}")
        raise


def monitor_job(job_id: str, account_id: str, check_interval: int = 30):
    """Monitor the fine-tuning job until completion using REST API.

    Args:
        job_id: The ID of the fine-tuning job
        account_id: The account ID
        check_interval: How often to check status (in seconds)
    """
    print(f"\n{'='*80}")
    print("STEP 3: MONITORING FINE-TUNING JOB")
    print(f"{'='*80}")

    print(f"Checking status every {check_interval} seconds...")
    print(f"(You can safely cancel this script and check status later)")

    try:
        while True:
            response = requests.get(
                f"{BASE_URL}/accounts/{account_id}/supervisedFineTuningJobs/{job_id}",
                headers=HEADERS
            )

            if response.status_code == 200:
                job = response.json()
                status = job.get('status', 'unknown')

                print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status}")

                if status == "succeeded" or status == "SUCCEEDED":
                    print(f"\n{'='*80}")
                    print("✅ FINE-TUNING COMPLETED SUCCESSFULLY!")
                    print(f"{'='*80}")
                    output_model = job.get('outputModel', job.get('outputModelId', 'unknown'))
                    print(f"Fine-tuned model ID: {output_model}")
                    print(f"\nYou can now use this model for inference:")
                    print(f"  Model ID: {output_model}")
                    print(f"\nTo use this model in your code:")
                    print(f"  from fireworks.client import Fireworks")
                    print(f"  client = Fireworks(api_key=FIREWORKS_API_KEY)")
                    print(f"  client.chat.completions.create(")
                    print(f"      model='{output_model}',")
                    print(f"      messages=[...]")
                    print(f"  )")
                    break

                elif status == "failed" or status == "FAILED":
                    print(f"\n❌ Fine-tuning failed!")
                    if 'error' in job:
                        print(f"Error: {job['error']}")
                    break

                elif status == "cancelled" or status == "CANCELLED":
                    print(f"\n⚠️  Fine-tuning was cancelled")
                    break

                else:
                    # Job is still running (queued, running, validating)
                    if 'trainedTokens' in job:
                        print(f"   Trained tokens: {job['trainedTokens']}")

                    time.sleep(check_interval)
            else:
                print(f"❌ Error checking job status: {response.status_code} - {response.text}")
                break

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Monitoring interrupted by user")
        print(f"Job is still running. You can check status later")
        print(f"  Job ID: {job_id}")
        print(f"  Account ID: {account_id}")
    except Exception as e:
        print(f"❌ Error monitoring job: {e}")
        raise


def main():
    """Main function to run the fine-tuning process."""
    print(f"\n{'='*80}")
    print("FIREWORKS.AI FINE-TUNING - CONTEXT AGGREGATION MODEL")
    print(f"{'='*80}\n")

    # Get account ID
    print("Getting account information...")
    account_id = get_account_id()
    print(f"✅ Account ID: {account_id}\n")

    # Generate dataset name
    dataset_name = f"{FINETUNE_JOB_NAME}-dataset"

    try:
        # Step 1: Upload dataset
        dataset_name = upload_dataset(TRAINING_DATA_PATH, dataset_name)

        # Step 2: Create fine-tuning job
        job_id = create_finetune_job(dataset_name, account_id)

        # Step 3: Monitor job (optional - can be cancelled)
        monitor_job(job_id, account_id)

    except Exception as e:
        print(f"\n❌ Fine-tuning process failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
