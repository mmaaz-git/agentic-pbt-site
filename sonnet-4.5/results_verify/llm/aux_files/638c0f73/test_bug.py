#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

print("Testing not_nulls function...")
print("=" * 50)

from llm.default_plugins.openai_models import not_nulls

# Test 1: Simple dictionary as claimed in bug report
print("\nTest 1: Simple dictionary {'a': 1}")
test_dict1 = {'a': 1}
try:
    result = not_nulls(test_dict1)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 2: Dictionary with None values
print("\nTest 2: Dictionary with None values {'a': 1, 'b': None, 'c': 'test'}")
test_dict2 = {'a': 1, 'b': None, 'c': 'test'}
try:
    result = not_nulls(test_dict2)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: Empty dictionary
print("\nTest 3: Empty dictionary {}")
test_dict3 = {}
try:
    result = not_nulls(test_dict3)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: What if we pass dict.items()?
print("\nTest 4: Passing dict.items() explicitly")
test_dict4 = {'a': 1, 'b': None, 'c': 'test'}
try:
    result = not_nulls(test_dict4.items())
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 5: List of tuples (what the current implementation expects)
print("\nTest 5: List of tuples [('a', 1), ('b', None), ('c', 'test')]")
test_data5 = [('a', 1), ('b', None), ('c', 'test')]
try:
    result = not_nulls(test_data5)
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")