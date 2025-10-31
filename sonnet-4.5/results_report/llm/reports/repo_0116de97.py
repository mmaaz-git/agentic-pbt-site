import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import remove_dict_none_values

# Test case from the bug report
input_dict = {"choices": [None, {"text": "response"}]}

result = remove_dict_none_values(input_dict)

print(f"Input:  {input_dict}")
print(f"Output: {result}")
print()

# Additional test cases to demonstrate the inconsistency
test_cases = [
    {"a": [None, 1]},
    {"b": [None]},
    {"c": [1, None, 2]},
    {"d": {"nested": None}},  # This should remove the None
    {"e": [{"nested": None}]},  # What happens here?
    {"": [None]},  # Minimal failing case
]

print("Additional test cases:")
print("=" * 50)
for test in test_cases:
    result = remove_dict_none_values(test)
    print(f"Input:  {test}")
    print(f"Output: {result}")
    print()