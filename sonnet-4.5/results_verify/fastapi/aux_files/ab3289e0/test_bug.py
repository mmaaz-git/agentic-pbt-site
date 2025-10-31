#!/usr/bin/env python3
"""Test the reported bug in is_union_of_base_models"""

from typing import Union, get_args, get_origin
from pydantic import BaseModel

# Import the function we're testing
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')
from fastapi.dependencies.utils import is_union_of_base_models

# First, let's test the hypothesis test case from the bug report
print("=== Testing the Hypothesis test case ===")
try:
    from hypothesis import given, strategies as st

    @given(st.just(Union))
    def test_empty_union_should_return_false(union_type):
        """An empty or unparameterized Union should not return True"""
        args = get_args(union_type)
        print(f"Testing with union_type: {union_type}")
        print(f"get_args() returns: {args}")
        print(f"Length of args: {len(args)}")

        if len(args) == 0:
            result = is_union_of_base_models(union_type)
            print(f"Result from is_union_of_base_models: {result}")
            assert result is False, "Empty Union should return False"

    # Run the test
    test_empty_union_should_return_false()
    print("Hypothesis test PASSED")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n=== Testing the reproduction example ===")
# The bug report's example - this should work normally
union_type = Union[str, int]
args = get_args(union_type)

print(f"Union type: {union_type}")
print(f"Args: {args}")
print(f"Length: {len(args)}")
print(f"Result: {is_union_of_base_models(union_type)}")

print("\n=== Testing edge cases ===")

# Test 1: Empty Union (if we can create one)
print("\n1. Testing bare Union:")
bare_union = Union
print(f"   Type: {bare_union}")
print(f"   get_args(): {get_args(bare_union)}")
print(f"   get_origin(): {get_origin(bare_union)}")
try:
    result = is_union_of_base_models(bare_union)
    print(f"   is_union_of_base_models result: {result}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Union with BaseModel types
print("\n2. Testing Union with BaseModel types:")
class Model1(BaseModel):
    x: int

class Model2(BaseModel):
    y: str

union_models = Union[Model1, Model2]
print(f"   Type: {union_models}")
print(f"   get_args(): {get_args(union_models)}")
result = is_union_of_base_models(union_models)
print(f"   is_union_of_base_models result: {result}")
print(f"   Expected: True (all are BaseModel subclasses)")

# Test 3: Union with mixed types
print("\n3. Testing Union with mixed types:")
union_mixed = Union[Model1, str]
print(f"   Type: {union_mixed}")
print(f"   get_args(): {get_args(union_mixed)}")
result = is_union_of_base_models(union_mixed)
print(f"   is_union_of_base_models result: {result}")
print(f"   Expected: False (not all are BaseModel subclasses)")

# Test 4: Non-Union type
print("\n4. Testing non-Union type:")
print(f"   Type: str")
result = is_union_of_base_models(str)
print(f"   is_union_of_base_models result: {result}")
print(f"   Expected: False (not a Union)")

# Test 5: Can we even create an empty Union?
print("\n5. Trying to create empty Union variations:")
try:
    empty1 = Union[()]
    print(f"   Union[()]: {empty1}, args: {get_args(empty1)}")
except Exception as e:
    print(f"   Union[()] failed: {e}")

try:
    # In Python, Union requires at least 2 arguments
    empty2 = eval("Union[]")
    print(f"   Union[]: {empty2}, args: {get_args(empty2)}")
except Exception as e:
    print(f"   Union[] failed: {e}")

# Test 6: Single-type Union (Python simplifies these)
print("\n6. Testing single-type Union:")
try:
    single_union = Union[str]
    print(f"   Union[str]: {single_union}")
    print(f"   Type after simplification: {type(single_union)}")
    print(f"   get_args(): {get_args(single_union)}")
    print(f"   get_origin(): {get_origin(single_union)}")
    result = is_union_of_base_models(single_union)
    print(f"   is_union_of_base_models result: {result}")
except Exception as e:
    print(f"   Error: {e}")