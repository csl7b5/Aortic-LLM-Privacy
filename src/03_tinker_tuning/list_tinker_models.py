import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

import os
import asyncio
from tinker.lib.public_interfaces.service_client import ServiceClient

async def main():
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
    client = ServiceClient()
    # Or just use the CLI to list models if available in Tinker. 
    # Since I don't know the exact API for listing, let's use the tinker CLI if one exists.
