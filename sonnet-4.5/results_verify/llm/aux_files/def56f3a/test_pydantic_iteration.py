#!/usr/bin/env python3
"""Test Pydantic BaseModel iteration behavior"""

from pydantic import BaseModel
from typing import Optional

# Create a simple Pydantic model similar to Options
class TestOptions(BaseModel):
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None

# Create an instance
options = TestOptions(temperature=0.7, max_tokens=None, seed=42)
print(f"TestOptions instance: {options}")
print(f"Type: {type(options)}")

# Test iteration behavior
print("\nIterating over TestOptions instance:")
for item in options:
    print(f"  Item: {item} (type: {type(item)})")

# Now test the not_nulls function with this
print("\nTesting dict comprehension like not_nulls:")
try:
    # This is what not_nulls does
    result = {key: value for key, value in options if value is not None}
    print(f"  Success! Result: {result}")
except Exception as e:
    print(f"  Error: {e}")

# Test with a regular dict
print("\nTesting with regular dict:")
regular_dict = {"temperature": 0.7, "max_tokens": None, "seed": 42}
print(f"Dict: {regular_dict}")

print("Iterating over dict (without .items()):")
for item in regular_dict:
    print(f"  Item: {item} (type: {type(item)})")

print("\nTrying dict comprehension on regular dict:")
try:
    result = {key: value for key, value in regular_dict if value is not None}
    print(f"  Success! Result: {result}")
except Exception as e:
    print(f"  Error: {e}")

print("\nWith .items():")
result = {key: value for key, value in regular_dict.items() if value is not None}
print(f"  Result: {result}")