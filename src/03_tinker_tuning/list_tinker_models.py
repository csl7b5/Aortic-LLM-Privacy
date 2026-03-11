"""
list_tinker_models.py
─────────────────────────────────────────────────────────────────────────────
Lists all recent Tinker fine-tuning runs and their model IDs, filtered to
show canary_Nx_12ep runs. Prints the model IDs you need to fill into
data/processed/training_datasets/canary_job_log.json.

Usage:
  python src/03_tinker_tuning/list_tinker_models.py

After filling in model IDs, run:
  python src/04_evaluation/generation/generate_canary_predictions.py
"""

import os
import sys
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.expanduser("~/tinker/tinker-cookbook"))

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

JOB_LOG_FILE = "data/processed/training_datasets/canary_job_log.json"
CANARY_RUNS  = ["canary_0x_12ep", "canary_1x_12ep", "canary_3x_12ep",
                "canary_5x_12ep", "canary_10x_12ep", "canary_25x_12ep"]


async def main():
    from tinker.lib.public_interfaces.service_client import ServiceClient

    client = ServiceClient()

    print("Fetching Tinker run list …\n")
    try:
        runs = await client.list_runs_async()
    except AttributeError:
        # Fallback: some versions expose list_models
        runs = await client.list_models_async()

    print(f"{'Run Name':<30}  {'Model ID / URI'}")
    print("-" * 90)

    canary_ids = {}   # run_name → model_id

    for run in runs:
        # Normalise across different SDK versions
        name     = getattr(run, "name",     None) or getattr(run, "run_name", str(run))
        model_id = getattr(run, "model_id", None) or getattr(run, "model_uri", None) or getattr(run, "uri", None)

        if any(tag in str(name) for tag in CANARY_RUNS):
            marker = " ← CANARY"
            canary_ids[str(name)] = str(model_id) if model_id else None
        else:
            marker = ""

        print(f"{str(name):<30}  {model_id}{marker}")

    print()

    if not canary_ids:
        print("No canary runs found yet. Training may still be in progress.")
        return

    # ── Auto-patch job log ─────────────────────────────────────────────────
    if os.path.exists(JOB_LOG_FILE):
        with open(JOB_LOG_FILE) as f:
            job_log = json.load(f)

        updated = 0
        for run_name, model_id in canary_ids.items():
            if run_name in job_log and model_id and job_log[run_name].get("model_id") is None:
                job_log[run_name]["model_id"] = model_id
                print(f"✓ Auto-filled model_id for {run_name}: {model_id}")
                updated += 1

        if updated:
            with open(JOB_LOG_FILE, "w") as f:
                json.dump(job_log, f, indent=2)
            print(f"\nUpdated {JOB_LOG_FILE} with {updated} model ID(s).")
        else:
            print("No new model IDs to fill in (all already present or training still running).")
    else:
        print(f"Job log not found at {JOB_LOG_FILE}. Run launch_canary_jobs.py first.")

    print("\nNext step:")
    print("  python src/04_evaluation/generation/generate_canary_predictions.py")


if __name__ == "__main__":
    asyncio.run(main())
