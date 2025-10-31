#!/usr/bin/env python3
"""Try to find if there's an actual way to trigger the alleged bug"""

from typing import Union, get_args, get_origin
from types import UnionType
from unittest.mock import Mock
from pydantic import BaseModel
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')
from fastapi.dependencies.utils import is_union_of_base_models

print("=== Attempting to create problematic Union types ===")

# Test 1: Mock object that behaves like empty Union
print("\n1. Creating a mock object that passes the origin check but has empty args:")
mock_union = Mock()
mock_union.__class__ = type(Union[int, str])

# Make get_origin return Union
original_get_origin = get_origin
def mock_get_origin(tp):
    if tp is mock_union:
        return Union
    return original_get_origin(tp)

# Make get_args return empty tuple
original_get_args = get_args
def mock_get_args(tp):
    if tp is mock_union:
        return ()
    return original_get_args(tp)

# Temporarily replace the functions
import typing
old_origin = typing.get_origin
old_args = typing.get_args
typing.get_origin = mock_get_origin
typing.get_args = mock_get_args

try:
    print(f"   Mock union: {mock_union}")
    print(f"   get_origin(mock_union): {get_origin(mock_union)}")
    print(f"   get_args(mock_union): {get_args(mock_union)}")

    # Now test the function with our mock
    from importlib import reload
    from fastapi.dependencies import utils
    reload(utils)

    result = utils.is_union_of_base_models(mock_union)
    print(f"   is_union_of_base_models(mock_union): {result}")
    print(f"   Expected (per bug report): True")
    print(f"   BUG CONFIRMED!" if result == True else "Bug not triggered")

finally:
    # Restore original functions
    typing.get_origin = old_origin
    typing.get_args = old_args

print("\n=== Testing with Python 3.10+ UnionType ===")
# In Python 3.10+, we have types.UnionType created by | operator
try:
    # Create union using | operator
    union_type_310 = int | str
    print(f"2. Testing int | str (UnionType):")
    print(f"   Type: {union_type_310}")
    print(f"   get_origin(): {get_origin(union_type_310)}")
    print(f"   get_args(): {get_args(union_type_310)}")

    result = is_union_of_base_models(union_type_310)
    print(f"   is_union_of_base_models result: {result}")

    # Try with BaseModel unions
    class Model1(BaseModel):
        x: int
    class Model2(BaseModel):
        y: str

    union_models_310 = Model1 | Model2
    print(f"\n3. Testing Model1 | Model2 (UnionType with BaseModels):")
    print(f"   Type: {union_models_310}")
    print(f"   get_origin(): {get_origin(union_models_310)}")
    print(f"   get_args(): {get_args(union_models_310)}")

    result = is_union_of_base_models(union_models_310)
    print(f"   is_union_of_base_models result: {result}")

except Exception as e:
    print(f"   UnionType test failed: {e}")

print("\n=== Manual verification of the logic flaw ===")
print("The alleged bug states that if we had:")
print("1. A type where get_origin() returns Union or UnionType")
print("2. But get_args() returns an empty tuple ()")
print("3. Then the function would return True instead of False")
print()
print("Let's trace through the code manually:")
print("def is_union_of_base_models(field_type):")
print("    origin = get_origin(field_type)  # Assume this returns Union")
print("    if origin is not Union and origin is not UnionType:")
print("        return False  # This would NOT execute")
print("    union_args = get_args(field_type)  # Assume this returns ()")
print("    for arg in union_args:  # Loop executes 0 times for empty tuple")
print("        if not lenient_issubclass(arg, BaseModel):")
print("            return False")
print("    return True  # <-- This WOULD be reached, returning True!")
print()
print("So the bug report IS CORRECT about the logic flaw:")
print("IF we could create a type where get_origin() returns Union/UnionType")
print("AND get_args() returns empty tuple, THEN it would incorrectly return True.")
print()
print("However, in practice with normal Python typing:")
print("- bare Union has get_origin() = None, so returns False early")
print("- Union[...] always has non-empty args")
print("- Python prevents creating Union[] or Union[()]")