import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

print("Testing with value 150 (out of range):")
try:
    opts = SharedOptions(logit_bias={"123": 150})
except ValidationError as e:
    print(f"Error: {e}")

print("\nTesting with value 50 (in range):")
try:
    opts = SharedOptions(logit_bias={"123": 50})
    print(f"Success! Created SharedOptions with logit_bias={opts.logit_bias}")
except ValidationError as e:
    print(f"Error: {e}")

print("\nTesting with invalid key (non-numeric):")
try:
    opts = SharedOptions(logit_bias={"abc": 50})
except ValidationError as e:
    print(f"Error: {e}")

print("\nTesting with value -150 (out of range, negative):")
try:
    opts = SharedOptions(logit_bias={"123": -150})
except ValidationError as e:
    print(f"Error: {e}")