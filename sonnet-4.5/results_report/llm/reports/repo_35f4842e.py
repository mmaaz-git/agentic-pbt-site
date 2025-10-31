#!/usr/bin/env python3
"""
Minimal reproduction case demonstrating the input mutation bug in redact_data()
"""
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import redact_data
import copy

# Create a test dictionary with image data
original = {
    "image_url": {"url": "data:image/png;base64,abc123"}
}

# Create a deep copy to preserve the original state
original_copy = copy.deepcopy(original)

# Call redact_data - this should not mutate the input
result = redact_data(original)

# Test 1: Check if the original was mutated (it shouldn't be)
print("Test 1: Check if original was mutated")
print(f"Original before: {original_copy}")
print(f"Original after:  {original}")
print(f"Are they equal? {original == original_copy}")
print()

# Test 2: Check what the original looks like now
print("Test 2: Verify the mutation")
print(f"Original is now: {original}")
print(f"Expected mutation: {{'image_url': {{'url': 'data:...'}}}}")
print(f"Matches expected mutation? {original == {'image_url': {'url': 'data:...'}}}")
print()

# Test 3: Check if the result is the same object as the input
print("Test 3: Check if result is same object as input")
print(f"Result is original? {result is original}")
print(f"Result id: {id(result)}, Original id: {id(original)}")