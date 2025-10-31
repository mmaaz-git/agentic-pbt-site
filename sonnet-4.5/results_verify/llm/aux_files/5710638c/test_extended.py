#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from typing import Optional

# Test with different iteration methods
class Options(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100

options = Options(temperature=0.5, max_tokens=200)

print("Testing different iteration methods on Pydantic BaseModel:")
print(f"Options: {options}")

# Method 1: Direct iteration
print("\n1. Direct iteration (for x in options):")
try:
    for item in options:
        print(f"  Item: {item}, Type: {type(item)}")
except Exception as e:
    print(f"  Error: {e}")

# Method 2: Trying to iterate like dict
print("\n2. Trying dict-style unpacking in comprehension:")
try:
    result = {key: value for key, value in options if value is not None}
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {e}")

# Method 3: Using .items() (doesn't exist on BaseModel)
print("\n3. Using .items():")
try:
    for key, value in options.items():
        print(f"  {key}: {value}")
except AttributeError as e:
    print(f"  AttributeError: BaseModel doesn't have .items() method")

# Method 4: Using model_dump().items()
print("\n4. Using model_dump().items():")
try:
    for key, value in options.model_dump().items():
        print(f"  {key}: {value}")
except Exception as e:
    print(f"  Error: {e}")

# Test the actual not_nulls function from the bug report
def not_nulls(data):
    return {key: value for key, value in data if value is not None}

print("\n5. Testing not_nulls function:")
try:
    result = not_nulls(options)
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {e}")

# Check if iteration behavior changed between versions
print("\n6. Checking BaseModel.__iter__ behavior:")
print(f"  hasattr(options, '__iter__'): {hasattr(options, '__iter__')}")
print(f"  BaseModel defines __iter__: {hasattr(BaseModel, '__iter__')}")

# Check what __iter__ returns
print("\n7. Manual iteration test:")
iterator = iter(options)
first_item = next(iterator)
print(f"  First item from iter(options): {first_item}")
print(f"  Type: {type(first_item)}")