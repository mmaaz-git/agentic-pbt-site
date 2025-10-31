import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

try:
    # Test with value above the allowed range
    opts = SharedOptions(logit_bias={"123": 150})
except ValidationError as e:
    print(f"Error for value 150: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value below the allowed range
    opts = SharedOptions(logit_bias={"456": -150})
except ValidationError as e:
    print(f"Error for value -150: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value at upper boundary (should work)
    opts = SharedOptions(logit_bias={"789": 100})
    print("Value 100: SUCCESS - No error raised")
except ValidationError as e:
    print(f"Error for value 100: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value at lower boundary (should work)
    opts = SharedOptions(logit_bias={"101": -100})
    print("Value -100: SUCCESS - No error raised")
except ValidationError as e:
    print(f"Error for value -100: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with invalid key (non-numeric)
    opts = SharedOptions(logit_bias={"abc": 50})
except ValidationError as e:
    print(f"Error for non-numeric key: {e}")