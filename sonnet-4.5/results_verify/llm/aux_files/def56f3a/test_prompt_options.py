#!/usr/bin/env python3
"""Test to understand Prompt.options structure"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

print("Checking Prompt class:")
print(f"Prompt type: {type(llm.Prompt)}")

print("\nChecking Options class:")
print(f"Options type: {type(llm.Options)}")
print(f"Options base classes: {llm.Options.__bases__}")

# Check if Options is a Pydantic model
try:
    from pydantic import BaseModel
    print(f"\nIs Options a Pydantic BaseModel? {issubclass(llm.Options, BaseModel)}")
except ImportError:
    print("Pydantic not available")

# Create an Options instance and test iteration
options = llm.Options(temperature=0.7, max_tokens=None, seed=42)
print(f"\nOptions instance: {options}")
print(f"Type of options: {type(options)}")

# Test iteration
print("\nTesting iteration over Options:")
try:
    for item in options:
        print(f"  Item: {item}")
        if isinstance(item, tuple):
            print(f"    Tuple length: {len(item)}")
except Exception as e:
    print(f"  Error: {e}")

# Test the not_nulls function with Options
from llm.default_plugins.openai_models import not_nulls

print("\nTesting not_nulls with Options instance:")
try:
    result = not_nulls(options)
    print(f"  Success! Result: {result}")
except Exception as e:
    print(f"  Error: {e}")