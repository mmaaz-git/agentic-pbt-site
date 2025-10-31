import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

# Test case that should work but crashes
test_dict = {'a': 1, 'b': None, 'c': 'test'}
print(f"Input: {test_dict}")
try:
    result = not_nulls(test_dict)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")