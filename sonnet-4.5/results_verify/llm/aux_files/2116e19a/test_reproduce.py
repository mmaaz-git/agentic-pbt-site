import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import remove_dict_none_values

# Test with the bug report's specific example
input_dict = {"choices": [None, {"text": "response"}]}
result = remove_dict_none_values(input_dict)
print(f"Input:  {input_dict}")
print(f"Output: {result}")
print(f"None values remain: {None in result.get('choices', [])}")

# Test with simpler case
simple_dict = {"a": [None, 1]}
simple_result = remove_dict_none_values(simple_dict)
print(f"\nSimple Input:  {simple_dict}")
print(f"Simple Output: {simple_result}")
print(f"None values remain: {None in simple_result.get('a', [])}")