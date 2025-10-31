import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')
from llm.default_plugins.openai_models import SharedOptions

# Test cases with various None scenarios
test_cases = [
    {"1": None},
    {"1": None, "2": 50},
    {"valid": "10", "invalid": None},
    {"100": None},
    {"a": None},  # Non-numeric key with None value
    {"1": "not_a_number"},  # This should raise ValueError
    {"1": 101},  # Out of range, should raise ValueError  
    {"1": -101},  # Out of range, should raise ValueError
]

for i, test_case in enumerate(test_cases):
    print(f"\nTest case {i+1}: {test_case}")
    try:
        options = SharedOptions(logit_bias=test_case)
        print(f"  Success: {options.logit_bias}")
    except TypeError as e:
        print(f"  TypeError: {e}")
    except ValueError as e:
        print(f"  ValueError: {e}")
