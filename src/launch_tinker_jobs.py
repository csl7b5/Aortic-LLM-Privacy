import os
import time
from tinker import TinkerClient

# Assume you have set up your API key via export TINKER_API_KEY="..."
client = TinkerClient(api_key=os.environ.get("TINKER_API_KEY"))

# Base Model
# Recommended baseline: Llama-3 (8B) or Mistral 7B. These are strong enough to follow 
# complex clinical prompt instructions but small enough that SFT memorization 
# attacks are highly effective and measurable.
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

# Training Dataset Paths (from the data prep pipeline)
M1_TRAIN_FILE = "../data/processed/tinker_train_M1_full.jsonl"
M2_TRAIN_FILE = "../data/processed/tinker_train_M2_coarsened.jsonl"

def upload_file(client, filepath, purpose="fine-tune"):
    """Helper to upload a local file to Tinker."""
    print(f"Uploading {filepath}...")
    response = client.files.create(
        file=open(filepath, "rb"),
        purpose=purpose
    )
    print(f"✅ Uploaded! File ID: {response.id}")
    return response.id

def launch_training():
    print("--- [1] Uploading Datasets ---")
    m1_file_id = upload_file(client, M1_TRAIN_FILE)
    m2_file_id = upload_file(client, M2_TRAIN_FILE)

    print("\n--- [2] Triggering M1 SFT Job (Full Identity) ---")
    # Training M1 on the fully identifiable records
    m1_job = client.fine_tuning.jobs.create(
        training_file=m1_file_id,
        model=BASE_MODEL,
        hyperparameters={
            "n_epochs": 3,      # Standard for SFT
            "batch_size": 4, 
            "learning_rate_multiplier": 2
        },
        suffix="M1_full_ft"
    )
    print(f"🚀 M1 Job launched. Job ID: {m1_job.id}")

    print("\n--- [3] Triggering M2 SFT Job (Coarsened Privacy) ---")
    # Training M2 on the privacy-mitigated records
    m2_job = client.fine_tuning.jobs.create(
        training_file=m2_file_id,
        model=BASE_MODEL,
        hyperparameters={
            "n_epochs": 3, 
            "batch_size": 4,
            "learning_rate_multiplier": 2
        },
        suffix="M2_coarsened_ft"
    )
    print(f"🚀 M2 Job launched. Job ID: {m2_job.id}")
    
    print("\nJobs are queued! You can check their status in the Tinker dashboard ")
    print("When they finish, you'll receive a fine_tuned_model ID for each.")

if __name__ == "__main__":
    launch_training()
