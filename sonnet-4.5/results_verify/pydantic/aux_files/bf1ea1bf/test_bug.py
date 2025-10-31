#!/usr/bin/env python3
"""Test to reproduce the reported bug with pydantic.experimental.pipeline constraints"""

from hypothesis import given, strategies as st, settings
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as

def count_validators(schema):
    """Count the number of validator functions in a schema"""
    if schema.get('type') == 'function-after':
        return 1 + count_validators(schema.get('schema', {}))
    if schema.get('type') == 'chain':
        return sum(count_validators(s) for s in schema.get('steps', []))
    if 'fields' in schema:
        for field in schema['fields'].values():
            if 'schema' in field:
                return count_validators(field['schema'])
    if 'schema' in schema:
        return count_validators(schema['schema'])
    return 0

print("Testing Property-Based Test from bug report...")
print("="*50)

@given(value=st.integers())
@settings(max_examples=10)
def test_constraint_validators_consistency(value):
    class ModelGt(BaseModel):
        v: Annotated[int, validate_as(int).gt(value)]

    class ModelGe(BaseModel):
        v: Annotated[int, validate_as(int).ge(value)]

    gt_validators = count_validators(ModelGt.__pydantic_core_schema__)
    ge_validators = count_validators(ModelGe.__pydantic_core_schema__)

    print(f"Value: {value}, Gt validators: {gt_validators}, Ge validators: {ge_validators}")

    assert gt_validators == ge_validators, \
        f"Gt has {gt_validators} validators but Ge has {ge_validators}"

try:
    test_constraint_validators_consistency()
    print("Property test passed!")
except AssertionError as e:
    print(f"Property test failed: {e}")

print("\n" + "="*50)
print("Testing specific reproduction example...")
print("="*50)

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(5)]

class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(5)]

gt_validators = count_validators(ModelGt.__pydantic_core_schema__)
ge_validators = count_validators(ModelGe.__pydantic_core_schema__)
lt_validators = count_validators(ModelLt.__pydantic_core_schema__)
le_validators = count_validators(ModelLe.__pydantic_core_schema__)

print(f"Gt constraint: {gt_validators} validator(s)")
print(f"Ge constraint: {ge_validators} validator(s)")
print(f"Lt constraint: {lt_validators} validator(s)")
print(f"Le constraint: {le_validators} validator(s)")

print("\n" + "="*50)
print("Testing actual validation behavior...")
print("="*50)

# Test that the constraints actually work correctly
test_cases = [
    (ModelGt, 6, True, "6 > 5"),
    (ModelGt, 5, False, "5 > 5"),
    (ModelGt, 4, False, "4 > 5"),
    (ModelGe, 6, True, "6 >= 5"),
    (ModelGe, 5, True, "5 >= 5"),
    (ModelGe, 4, False, "4 >= 5"),
    (ModelLt, 4, True, "4 < 5"),
    (ModelLt, 5, False, "5 < 5"),
    (ModelLt, 6, False, "6 < 5"),
    (ModelLe, 4, True, "4 <= 5"),
    (ModelLe, 5, True, "5 <= 5"),
    (ModelLe, 6, False, "6 <= 5"),
]

for model_class, test_value, should_pass, description in test_cases:
    try:
        instance = model_class(value=test_value)
        if should_pass:
            print(f"✓ {description}: correctly accepted")
        else:
            print(f"✗ {description}: incorrectly accepted (should have failed)")
    except Exception as e:
        if not should_pass:
            print(f"✓ {description}: correctly rejected")
        else:
            print(f"✗ {description}: incorrectly rejected (should have passed)")
            print(f"  Error: {e}")

print("\n" + "="*50)
print("Examining schema structure for Gt...")
print("="*50)
import json
print(json.dumps(ModelGt.__pydantic_core_schema__, indent=2))

print("\n" + "="*50)
print("Examining schema structure for Ge...")
print("="*50)
print(json.dumps(ModelGe.__pydantic_core_schema__, indent=2))