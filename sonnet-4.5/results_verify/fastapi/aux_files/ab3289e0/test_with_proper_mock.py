#!/usr/bin/env python3
"""Test with a properly crafted mock to trigger the bug"""

from typing import Union, get_args, get_origin
from types import UnionType
from pydantic import BaseModel

print("=== Creating a custom type that mimics empty Union ===")

# Create a custom class that behaves like a Union but with empty args
class EmptyUnionMock:
    """A type that pretends to be a Union with no arguments"""
    __origin__ = Union
    __args__ = ()

    def __repr__(self):
        return "EmptyUnionMock(origin=Union, args=())"

empty_union = EmptyUnionMock()

print(f"Custom type: {empty_union}")
print(f"get_origin(): {get_origin(empty_union)}")
print(f"get_args(): {get_args(empty_union)}")
print(f"get_origin() is Union: {get_origin(empty_union) is Union}")
print(f"get_origin() is UnionType: {get_origin(empty_union) is UnionType}")

print("\n=== Testing is_union_of_base_models with the mock ===")

# Import and test the function
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')
from fastapi.dependencies.utils import is_union_of_base_models

result = is_union_of_base_models(empty_union)
print(f"is_union_of_base_models(empty_union): {result}")
print(f"Expected per bug report: True (incorrectly)")
print(f"Expected correct behavior: False")
print()
if result == True:
    print("*** BUG CONFIRMED! ***")
    print("The function incorrectly returns True for a Union with no arguments.")
    print("The bug report is VALID!")
else:
    print("The function correctly returns False.")
    print("Bug not triggered with this mock.")

print("\n=== Alternative test with UnionType mock ===")

class EmptyUnionTypeMock:
    """A type that pretends to be a UnionType with no arguments"""
    __origin__ = UnionType
    __args__ = ()

    def __repr__(self):
        return "EmptyUnionTypeMock(origin=UnionType, args=())"

empty_union_type = EmptyUnionTypeMock()

print(f"Custom type: {empty_union_type}")
print(f"get_origin(): {get_origin(empty_union_type)}")
print(f"get_args(): {get_args(empty_union_type)}")

result2 = is_union_of_base_models(empty_union_type)
print(f"is_union_of_base_models(empty_union_type): {result2}")

if result2 == True:
    print("*** BUG CONFIRMED with UnionType! ***")
else:
    print("Function correctly returns False for UnionType mock.")

print("\n=== Summary ===")
print("The bug report claims that if a type has:")
print("  1. get_origin() returning Union or UnionType")
print("  2. get_args() returning empty tuple ()")
print("Then is_union_of_base_models would incorrectly return True.")
print()
print(f"Testing shows:")
print(f"  - With Union origin and empty args: returns {result}")
print(f"  - With UnionType origin and empty args: returns {result2}")
if result == True or result2 == True:
    print("\nThe bug IS REAL and can be triggered with crafted types!")
else:
    print("\nThe bug could not be triggered even with crafted types.")