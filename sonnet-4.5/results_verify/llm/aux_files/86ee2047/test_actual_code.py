#!/usr/bin/env python3
"""Test with the actual SharedOptions from the OpenAI models plugin"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions, not_nulls

# Test 1: Test with the actual SharedOptions class
print("=== Testing with actual SharedOptions from llm ===")
print(f"SharedOptions class: {SharedOptions}")

# Create an instance with some values
opts = SharedOptions(temperature=0.5, max_tokens=100)
print(f"\nCreated options: {opts}")
print(f"Type: {type(opts)}")

# Test the actual not_nulls function
print("\n=== Testing actual not_nulls function ===")
try:
    result = not_nulls(opts)
    print(f"Result: {result}")
    print("SUCCESS: not_nulls worked without error!")
except ValueError as e:
    print(f"FAILED with ValueError: {e}")
except Exception as e:
    print(f"FAILED with unexpected error: {type(e).__name__}: {e}")

# Check how the actual SharedOptions iterates
print("\n=== How does SharedOptions iterate? ===")
opts = SharedOptions(temperature=0.7, max_tokens=200, seed=None, top_p=0.9)
print("Direct iteration over opts:")
for item in opts:
    print(f"  {item} (type: {type(item)})")

# Test unpacking
print("\nUnpacking during iteration:")
try:
    for key, value in opts:
        print(f"  key={key}, value={value}")
except ValueError as e:
    print(f"  ValueError: {e}")

# Test with None values
print("\n=== Testing with None values ===")
opts = SharedOptions(temperature=None, max_tokens=100, seed=None)
print(f"Options with None values: {opts}")
try:
    result = not_nulls(opts)
    print(f"Result (filtered): {result}")
    print("Correctly filtered None values!")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")