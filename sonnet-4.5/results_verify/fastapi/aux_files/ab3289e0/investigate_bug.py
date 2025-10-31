#!/usr/bin/env python3
"""Deep investigation of the alleged bug"""

from typing import Union, get_args, get_origin
from pydantic import BaseModel
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')
from fastapi.dependencies.utils import is_union_of_base_models

print("=== Analyzing the bare Union type ===")
bare_union = Union
print(f"bare_union = Union")
print(f"Type: {bare_union}")
print(f"get_origin(bare_union): {get_origin(bare_union)}")
print(f"get_args(bare_union): {get_args(bare_union)}")

print("\n=== Step-by-step execution of is_union_of_base_models(Union) ===")

# Replicate the function logic
from fastapi.types import UnionType

field_type = Union
print(f"1. field_type = Union")

origin = get_origin(field_type)
print(f"2. origin = get_origin(field_type) = {origin}")

print(f"3. Checking: if origin is not Union and origin is not UnionType:")
print(f"   origin is {origin}")
print(f"   Union is {Union}")
print(f"   UnionType is {UnionType}")
print(f"   origin is not Union: {origin is not Union}")
print(f"   origin is not UnionType: {origin is not UnionType}")
print(f"   Combined condition: {origin is not Union and origin is not UnionType}")

if origin is not Union and origin is not UnionType:
    print("4. Would return False here")
else:
    print("4. Did not return False, continuing...")

    union_args = get_args(field_type)
    print(f"5. union_args = get_args(field_type) = {union_args}")
    print(f"   Length of union_args: {len(union_args)}")

    print("6. Entering for loop:")
    found_non_base_model = False
    for i, arg in enumerate(union_args):
        print(f"   Iteration {i}: checking arg {arg}")
        # This loop won't execute if union_args is empty
        found_non_base_model = True
        break

    if not found_non_base_model and len(union_args) == 0:
        print("   Loop did not execute (empty union_args)")
        print("7. Would return True here (this is the alleged bug)")
    else:
        print("   Loop executed")

print("\n=== Checking what get_origin returns for bare Union ===")
print(f"Union type object: {Union}")
print(f"get_origin(Union): {get_origin(Union)}")
print(f"Union is Union: {Union is Union}")
print(f"get_origin(Union) is Union: {get_origin(Union) is Union}")
print(f"get_origin(Union) is None: {get_origin(Union) is None}")

print("\n=== The key insight ===")
print("For bare Union (just typing.Union without parameters):")
print(f"  get_origin(Union) returns: {get_origin(Union)}")
print(f"  This is None, not Union itself")
print(f"  So the check 'origin is not Union' evaluates to: {get_origin(Union) is not Union}")
print(f"  And 'origin is not UnionType' evaluates to: {get_origin(Union) is not UnionType}")
print(f"  Combined: {(get_origin(Union) is not Union) and (get_origin(Union) is not UnionType)}")
print("  Therefore, the function returns False early, not True!")

print("\n=== Testing the actual function ===")
result = is_union_of_base_models(Union)
print(f"is_union_of_base_models(Union) = {result}")
print(f"Bug claim: Should return False for empty Union")
print(f"Actual behavior: Returns {result}")
print(f"Bug claim is: {'CORRECT' if result == False else 'INCORRECT'}")