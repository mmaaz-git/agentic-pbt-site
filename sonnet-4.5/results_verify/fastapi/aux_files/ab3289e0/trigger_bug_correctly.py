#!/usr/bin/env python3
"""Properly trigger the bug by creating a type that get_origin recognizes"""

from typing import Union, get_args, get_origin, _BaseGenericAlias
from types import UnionType
from pydantic import BaseModel

print("=== Creating a type that properly triggers the bug ===")

# Create a custom generic alias that has Union as origin but empty args
class EmptyUnionAlias(_BaseGenericAlias):
    """A Union-like type with no arguments"""
    def __init__(self):
        self.__origin__ = Union
        self.__args__ = ()
        # Other attributes to make it work properly
        self._inst = False
        self._name = 'EmptyUnion'

    def __repr__(self):
        return f"EmptyUnionAlias(origin={self.__origin__}, args={self.__args__})"

empty_union = EmptyUnionAlias()

print(f"Custom type: {empty_union}")
print(f"isinstance of _BaseGenericAlias: {isinstance(empty_union, _BaseGenericAlias)}")
print(f"get_origin(): {get_origin(empty_union)}")
print(f"get_args(): {get_args(empty_union)}")
print(f"get_origin() is Union: {get_origin(empty_union) is Union}")

print("\n=== Testing is_union_of_base_models with this type ===")

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')
from fastapi.dependencies.utils import is_union_of_base_models

result = is_union_of_base_models(empty_union)
print(f"is_union_of_base_models(empty_union): {result}")
print()

if result == True:
    print("*** BUG CONFIRMED! ***")
    print("The function incorrectly returns True for a Union with empty arguments!")
    print("This violates the docstring: 'Check if field type is a Union where all members are BaseModel subclasses.'")
    print("An empty Union has no members, so it cannot have 'all members' be BaseModel subclasses.")
else:
    print("The function returned False.")

print("\n=== Analysis ===")
print("Let's trace through the logic manually:")
print(f"1. origin = get_origin(empty_union) = {get_origin(empty_union)}")
print(f"2. origin is not Union: {get_origin(empty_union) is not Union}")
print(f"3. origin is not UnionType: {get_origin(empty_union) is not UnionType}")
print(f"4. Combined check (should return False early): {(get_origin(empty_union) is not Union) and (get_origin(empty_union) is not UnionType)}")

if not ((get_origin(empty_union) is not Union) and (get_origin(empty_union) is not UnionType)):
    print("5. Did NOT return False early, continuing...")
    union_args = get_args(empty_union)
    print(f"6. union_args = {union_args}")
    print(f"7. Loop over union_args (length={len(union_args)}):")
    if len(union_args) == 0:
        print("   Loop body never executes (empty tuple)")
        print("8. Function reaches 'return True'")
        print("   This is the BUG!")