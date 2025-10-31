#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from typing import Optional

class Options(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100

def not_nulls(data):
    return {key: value for key, value in data if value is not None}

# Test Case 1: With non-None values
print("Test 1: Creating Options with max_tokens=50")
options = Options(max_tokens=50)
print(f"Options instance: {options}")
print(f"Type: {type(options)}")

# Try the not_nulls function
print("\nTrying not_nulls function...")
try:
    result = not_nulls(options)
    print(f"Success! Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Test Case 2: Iteration behavior
print("\n\nTest 2: Understanding Pydantic v2 iteration")
options2 = Options(temperature=0.7, max_tokens=100)
print(f"Options instance: {options2}")
print("Iterating over options2:")
for item in options2:
    print(f"  Item: {item}, Type: {type(item)}")

print("\nIterating with model_dump().items():")
for key, value in options2.model_dump().items():
    print(f"  {key}: {value}")