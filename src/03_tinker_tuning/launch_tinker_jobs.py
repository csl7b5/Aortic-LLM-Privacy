import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import os
import sys
import asyncio

# Dynamically inject the local cookbook into path since it is not pip installed
sys.path.append(os.path.expanduser("~/tinker/tinker-cookbook"))

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

# Attempt to load from environment first, then config
api_key = os.environ.get("TINKER_API_KEY")
if not api_key:
    try:
        from config import TINKER_API_KEY
        api_key = TINKER_API_KEY
    except ImportError:
        pass
        
if not api_key:
    print("ERROR: TINKER_API_KEY not found in environment variables or config.py!")
    print("Please export TINKER_API_KEY='your_key' or add it to src/utils/config.py.")
    sys.exit(1)
    
os.environ["TINKER_API_KEY"] = api_key

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def run_job(run_name: str, dataset_path: str, epochs: int = 3):
    print(f"\n=============================================")
    print(f"Launching Tinker Job: {run_name} ({epochs} Epochs)")
    print(f"Dataset: {dataset_path}")
    print(f"=============================================")
    
    # 1. Setup Common Config for Conversation Parsing
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=BASE_MODEL,
        renderer_name=renderer_name,
        max_length=4096,  # Cards aren't extremely long
        batch_size=4,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    
    # 2. Setup the specific dataset
    dataset = FromConversationFileBuilder(
        common_config=common_config, 
        file_path=dataset_path
    )
    
    # 3. Create the Training Blueprint
    blueprint = chz.Blueprint(train.Config).apply({
        "log_path": f"/tmp/tinker/aortic_memorization_{run_name}",
        "model_name": BASE_MODEL,
        "dataset_builder": dataset,
        "learning_rate": 2e-5,
        "lr_schedule": "cosine",
        "num_epochs": epochs,
        "eval_every": 10,
    })
    
    # 4. Generate the config and run
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))

def launch_training():
    from config import OUT_M1_EXACT_PATH, OUT_M2_PATH
    
    # Execute Phase II M1 Job (Exact Age - 12 Epochs)
    m1_exact_path = os.path.abspath(OUT_M1_EXACT_PATH)
    run_job("M1_Exact_Age_12_Epochs", m1_exact_path, epochs=12)
    
    # Execute Phase II M2 Job (Coarsened - 12 Epochs)
    m2_path = os.path.abspath(OUT_M2_PATH)
    run_job("M2_Coarsened_12_Epochs", m2_path, epochs=12)

if __name__ == "__main__":
    launch_training()
