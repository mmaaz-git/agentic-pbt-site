#!/usr/bin/env python3
"""Test what prompt.options actually is"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.models import Prompt, Options
from pydantic import BaseModel

# Create an Options instance
class TestOptions(Options):
    temperature: float = 0.7
    max_tokens: int = 100

# Test 1: Prompt created with None options
print("Test 1: Prompt with None options")
p1 = Prompt(prompt="test", model=None, options=None)
print(f"  p1.options: {p1.options}")
print(f"  type(p1.options): {type(p1.options)}")
print(f"  isinstance(p1.options, dict): {isinstance(p1.options, dict)}")

print()

# Test 2: Prompt created with dict options
print("Test 2: Prompt with dict options")
p2 = Prompt(prompt="test", model=None, options={"temperature": 0.7, "max_tokens": None})
print(f"  p2.options: {p2.options}")
print(f"  type(p2.options): {type(p2.options)}")
print(f"  isinstance(p2.options, dict): {isinstance(p2.options, dict)}")

print()

# Test 3: Prompt created with Options instance
print("Test 3: Prompt with Options instance")
opts = TestOptions()
p3 = Prompt(prompt="test", model=None, options=opts)
print(f"  p3.options: {p3.options}")
print(f"  type(p3.options): {type(p3.options)}")
print(f"  isinstance(p3.options, dict): {isinstance(p3.options, dict)}")
print(f"  isinstance(p3.options, BaseModel): {isinstance(p3.options, BaseModel)}")

# Try iterating over the options
print("\nIterating over p3.options (BaseModel):")
try:
    for item in p3.options:
        print(f"  Item: {item}")
except Exception as e:
    print(f"  ERROR iterating: {e}")

# Convert to dict
print(f"\ndict(p3.options): {dict(p3.options)}")

# Now test iterating over dict(p3.options)
print("\nIterating over dict(p3.options):")
for item in dict(p3.options):
    print(f"  Item: {item}")