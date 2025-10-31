import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import redact_data
import copy

original = {
    "image_url": {"url": "data:image/png;base64,abc123"}
}

original_copy = copy.deepcopy(original)
result = redact_data(original)

print(f"Original before: {original_copy}")
print(f"Original after: {original}")
print(f"Result: {result}")
print(f"Original != original_copy: {original != original_copy}")
expected = {'image_url': {'url': 'data:...'}}
print(f"Original == expected: {original == expected}")
print(f"Result is original: {result is original}")