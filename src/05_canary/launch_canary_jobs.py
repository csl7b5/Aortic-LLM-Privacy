import os
import sys
import asyncio
import argparse
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))
sys.path.append(os.path.expanduser("~/tinker/tinker-cookbook"))

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

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

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EPOCHS = 12
INJECTION_COUNTS = [0, 1, 2, 3, 5, 10, 25]

def get_canary_config(canary_id):
    if canary_id == 1:
        return {
            "train_dir": "data/processed/training_datasets",
            "log_file": "data/processed/training_datasets/canary_job_log.json",
            "prefix": "canary"
        }
    else:
        return {
            "train_dir": "data/processed/training_datasets/canary2",
            "log_file": "data/processed/training_datasets/canary2_job_log.json",
            "prefix": "canary2"
        }

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
        "log_path":      f"/tmp/tinker/{run_name}",
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


def main(canary_id, counts):
    config = get_canary_config(canary_id)
    
    job_log = {}
    if os.path.exists(config["log_file"]):
        with open(config["log_file"]) as f:
            job_log = json.load(f)

    counts_to_run = [c for c in counts if c in INJECTION_COUNTS]
    print(f"Will launch {len(counts_to_run)} canary job(s): {counts_to_run}")

    for n in counts_to_run:
        run_name = f"{config['prefix']}_{n}x_12ep"
        dataset_path = os.path.abspath(
            os.path.join(config["train_dir"], f"tinker_train_M1_{config['prefix']}_{n}x.jsonl")
        )

        if not os.path.exists(dataset_path):
            print(f"\n⚠ Training file not found: {dataset_path}")
            continue

        if run_name in job_log:
            print(f"\n⏭  Skipping {run_name} (already in job log)")
            continue

        run_job(run_name, dataset_path, epochs=EPOCHS)

        job_log[run_name] = {
            "injection_count": n,
            "dataset":         dataset_path,
            "epochs":          EPOCHS,
            "model_id":        None,
        }
        os.makedirs(os.path.dirname(config["log_file"]), exist_ok=True)
        with open(config["log_file"], "w") as f:
            json.dump(job_log, f, indent=2)

    print(f"\n{'='*55}")
    print("All canary jobs launched.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, choices=[1, 2], required=True, help="Canary ID (1 or 2)")
    parser.add_argument("--counts", nargs="+", type=int, default=INJECTION_COUNTS)
    args = parser.parse_args()
    main(args.id, args.counts)
