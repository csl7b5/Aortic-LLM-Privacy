"""
launch_canary_jobs.py
─────────────────────────────────────────────────────────────────────────────
Launches 6 Tinker fine-tuning jobs for the canary memorization threshold
experiment — one job per injection frequency (0×, 1×, 3×, 5×, 10×, 25×).

Each job trains on the full M1 dataset + N copies of the canary patient,
using identical hyperparameters to the original M1 training (12 epochs).

After jobs complete, retrieve model IDs using:
  python src/03_tinker_tuning/list_tinker_models.py

Then fill in the MODEL_IDS dict in:
  src/04_evaluation/generation/generate_canary_predictions.py

Usage:
  python src/03_tinker_tuning/launch_canary_jobs.py
  python src/03_tinker_tuning/launch_canary_jobs.py --counts 1 5 10  # subset
"""

import os
import sys
import asyncio
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.expanduser("~/tinker/tinker-cookbook"))

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# ── Auth ──────────────────────────────────────────────────────────────────────
api_key = os.environ.get("TINKER_API_KEY")
if not api_key:
    try:
        from config import TINKER_API_KEY
        api_key = TINKER_API_KEY
    except ImportError:
        pass

if not api_key:
    print("ERROR: TINKER_API_KEY not found in environment or config.py")
    sys.exit(1)

os.environ["TINKER_API_KEY"] = api_key

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "meta-llama/Llama-3.1-8B-Instruct"
EPOCHS           = 12          # Same as original M1
INJECTION_COUNTS = [0, 1, 3, 5, 10, 25]
TRAIN_DIR        = "data/processed/training_datasets"
LOG_FILE         = "data/processed/training_datasets/canary_job_log.json"


def run_job(run_name: str, dataset_path: str, epochs: int = 12):
    print(f"\n{'='*55}")
    print(f"Launching: {run_name}")
    print(f"Dataset:   {dataset_path}")
    print(f"Epochs:    {epochs}")
    print(f"{'='*55}")

    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=dataset_path,
    )
    blueprint = chz.Blueprint(train.Config).apply({
        "log_path":      f"/tmp/tinker/canary_{run_name}",
        "model_name":    BASE_MODEL,
        "dataset_builder": dataset,
        "learning_rate": 2e-5,
        "lr_schedule":   "cosine",
        "num_epochs":    epochs,
        "eval_every":    10,
    })
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))
    print(f"\n✓ Job complete: {run_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts", nargs="+", type=int,
        default=INJECTION_COUNTS,
        help="Which injection counts to run (default: all)"
    )
    args = parser.parse_args()

    # Load existing log if present
    job_log = {}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            job_log = json.load(f)

    counts_to_run = [c for c in args.counts if c in INJECTION_COUNTS]
    print(f"Will launch {len(counts_to_run)} canary job(s): {counts_to_run}")

    for n in counts_to_run:
        run_name = f"canary_{n}x_12ep"
        dataset_path = os.path.abspath(
            os.path.join(TRAIN_DIR, f"tinker_train_M1_canary_{n}x.jsonl")
        )

        if not os.path.exists(dataset_path):
            print(f"\n⚠ Training file not found: {dataset_path}")
            print("  Run create_canary_data.py first.")
            continue

        if run_name in job_log:
            print(f"\n⏭  Skipping {run_name} (already in job log)")
            continue

        run_job(run_name, dataset_path, epochs=EPOCHS)

        # Log completion (model ID to be filled in manually after listing models)
        job_log[run_name] = {
            "injection_count": n,
            "dataset":         dataset_path,
            "epochs":          EPOCHS,
            "model_id":        None,  # fill after list_tinker_models.py
        }
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "w") as f:
            json.dump(job_log, f, indent=2)

    print(f"\n{'='*55}")
    print("All canary jobs launched.")
    print(f"Job log: {LOG_FILE}")
    print()
    print("Next steps:")
    print("  1. python src/03_tinker_tuning/list_tinker_models.py")
    print("     → Find the model IDs for each canary_Nx_12ep run")
    print("  2. Fill MODEL_IDS in generate_canary_predictions.py")
    print("  3. python src/04_evaluation/generation/generate_canary_predictions.py")
    print("  4. python src/04_evaluation/analysis/evaluate_canary.py")


if __name__ == "__main__":
    main()
