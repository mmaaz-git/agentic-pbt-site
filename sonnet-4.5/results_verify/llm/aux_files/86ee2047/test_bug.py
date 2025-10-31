#!/usr/bin/env python3
"""Test script to reproduce the not_nulls bug"""

from pydantic import BaseModel, Field
from typing import Optional

# First, let's reproduce the exact function from the codebase
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}

# Define the SharedOptions model similar to the one in the codebase
class SharedOptions(BaseModel):
    temperature: Optional[float] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    seed: Optional[int] = Field(default=42)

# Test 1: Try the bug report's failing case
print("=== Test 1: Basic reproduction ===")
try:
    opts = SharedOptions(temperature=0.5, max_tokens=100)
    print(f"Created options: {opts}")
    print(f"Type of opts: {type(opts)}")

    # This should fail according to the bug report
    result = not_nulls(opts)
    print(f"Result: {result}")
    print("TEST 1 PASSED - No error occurred!")
except ValueError as e:
    print(f"TEST 1 FAILED with ValueError: {e}")
except Exception as e:
    print(f"TEST 1 FAILED with unexpected error: {type(e).__name__}: {e}")

# Test 2: Let's understand how Pydantic models iterate
print("\n=== Test 2: Understanding Pydantic iteration ===")
opts = SharedOptions(temperature=0.5, max_tokens=100)
print("Iterating over Pydantic model directly:")
for item in opts:
    print(f"  Item: {item} (type: {type(item)})")

print("\nIterating over Pydantic model with model_dump():")
for key, value in opts.model_dump().items():
    print(f"  Key: {key}, Value: {value}")

# Test 3: Try the proposed fix
print("\n=== Test 3: Testing proposed fix ===")
def not_nulls_fixed(data) -> dict:
    return {key: value for key, value in data.model_dump().items() if value is not None}

try:
    opts = SharedOptions(temperature=0.5, max_tokens=100, seed=None)
    result_fixed = not_nulls_fixed(opts)
    print(f"Fixed function result: {result_fixed}")
    print("TEST 3 PASSED - Fixed version works!")
except Exception as e:
    print(f"TEST 3 FAILED: {type(e).__name__}: {e}")

# Test 4: Run the Hypothesis test
print("\n=== Test 4: Running Hypothesis test ===")
try:
    from hypothesis import given, strategies as st

    @given(
        st.one_of(st.none(), st.floats(min_value=0, max_value=2)),
        st.one_of(st.none(), st.integers(min_value=1)),
    )
    def test_not_nulls_filters_none_values(temperature, max_tokens):
        opts = SharedOptions(temperature=temperature, max_tokens=max_tokens)
        result = not_nulls(opts)

        for key, value in result.items():
            assert value is not None

    # Try to run the test
    test_not_nulls_filters_none_values()
    print("TEST 4 PASSED - Hypothesis test completed without errors")
except Exception as e:
    print(f"TEST 4 FAILED: {type(e).__name__}: {e}")

# Test 5: Check what happens when iterating and unpacking
print("\n=== Test 5: Direct unpacking test ===")
opts = SharedOptions(temperature=0.5, max_tokens=100)
print("Attempting to unpack during iteration:")
try:
    for key, value in opts:
        print(f"  Key: {key}, Value: {value}")
except ValueError as e:
    print(f"  ValueError occurred: {e}")
    print(f"  This confirms the bug - can't unpack field names into two variables")