import os
import sys
import json
import asyncio
import argparse
import tenacity
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'utils')))
sys.path.append(os.path.expanduser("~/tinker/tinker-cookbook"))

import tinker
from tinker.lib.public_interfaces.service_client import ServiceClient
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

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
N_GENERATIONS = 20
TEMPERATURE = 1.0
MAX_TOKENS = 600
INJECTION_COUNTS = [0, 1, 2, 3, 5, 10, 25]

def get_canary_config(canary_id):
    if canary_id == 1:
        return {
            "eval_prompt": Path("data/processed/eval_prompts/eval_prompts_canary.jsonl"),
            "job_log": Path("data/processed/training_datasets/canary_job_log.json"),
            "out_dir": Path("data/results/predictions/canary"),
            "file_prefix": "canary"
        }
    else:
        return {
            "eval_prompt": Path("data/processed/eval_prompts/eval_prompts_canary2.jsonl"),
            "job_log": Path("data/processed/training_datasets/canary2_job_log.json"),
            "out_dir": Path("data/results/predictions/canary2"),
            "file_prefix": "canary2"
        }

@tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=60), stop=tenacity.stop_after_attempt(5), reraise=True)
async def sample_with_retry(sampling_client, minput, num_samples, stop_condition):
    return await sampling_client.sample_async(
        minput,
        num_samples=num_samples,
        sampling_params=tinker.SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS, stop=stop_condition),
    )

def load_model_ids(config):
    if not config["job_log"].exists():
        print(f"ERROR: Job log not found at {config['job_log']}")
        sys.exit(1)

    with open(config["job_log"]) as f:
        job_log = json.load(f)

    model_ids = {}
    for run_name, info in job_log.items():
        n = info["injection_count"]
        model_ids[n] = info.get("model_id")

    for n in INJECTION_COUNTS:
        if n not in model_ids: model_ids[n] = None
    return model_ids

async def generate_for_model(injection_count, model_id, prompt, out_path):
    client = ServiceClient()
    sampling_client = client.create_sampling_client(model_path=model_id)
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer  = get_renderer("llama3", tokenizer=tokenizer)
    stop_cond = renderer.get_stop_sequences()

    messages = [{"role": "user", "content": prompt["prompt_text"]}]
    minput   = renderer.build_generation_prompt(messages)

    print(f"  Generating {N_GENERATIONS} samples for {injection_count}x model ({model_id}) …")
    result = await sample_with_retry(sampling_client, minput, N_GENERATIONS, stop_cond)

    generations = []
    for seq in result.sequences:
        parsed, _ = renderer.parse_response(seq.tokens)
        generations.append(parsed["content"])

    record = {
        "injection_count": injection_count,
        "model_id": model_id,
        "patient_id": prompt["patient_id"],
        "prompt_id": prompt["prompt_id"],
        "generations": generations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: f.write(json.dumps(record) + "\n")
    print(f"  ✓ Wrote -> {out_path}  ({len(generations)} generations)")

async def main(canary_id):
    config = get_canary_config(canary_id)
    model_ids = load_model_ids(config)
    
    with open(config["eval_prompt"]) as f:
        prompt = json.loads(f.readline())

    none_count = sum(1 for v in model_ids.values() if v is None)
    if none_count == len(INJECTION_COUNTS):
        print("ERROR: All model IDs are None.")
        sys.exit(1)

    for n in INJECTION_COUNTS:
        model_id = model_ids.get(n)
        if model_id is None:
            print(f"Skipping {n}x - no model ID configured")
            continue

        out_path = config["out_dir"] / f"{config['file_prefix']}_{n}x_predictions.jsonl"
        if out_path.exists():
            print(f"Already exists, skipping: {out_path}")
            continue

        await generate_for_model(n, model_id, prompt, out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, choices=[1, 2], required=True, help="Canary ID (1 or 2)")
    args = parser.parse_args()
    asyncio.run(main(args.id))
