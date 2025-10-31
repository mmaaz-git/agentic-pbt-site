#!/usr/bin/env python3
"""Test the reported bug in pydantic.experimental.pipeline not_in constraint"""

from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform
from typing import Annotated
import traceback

print("=" * 60)
print("Testing the basic reproduction case")
print("=" * 60)

try:
    class Model(BaseModel):
        field: Annotated[int, transform(lambda x: x).not_in([1, 2, 3])]

    # This should raise ValidationError according to bug report
    result = Model(field=2)
    print(f"✓ Model accepted value: {result.field}")
    print("  -> Bug claim: This should have been rejected")
except ValidationError as e:
    print(f"✗ ValidationError raised: {e}")
    print("  -> Value was correctly rejected")
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing with property-based test approach")
print("=" * 60)

# Simplified version of the hypothesis test
test_cases = [
    (1, [1, 2, 3]),  # value in list
    (2, [1, 2, 3]),  # value in list
    (4, [1, 2, 3]),  # value not in list
    (0, [0]),         # edge case: zero
    (-1, [-1, 0, 1]), # negative number
]

for value, excluded_values in test_cases:
    try:
        class TestModel(BaseModel):
            field: Annotated[int, transform(lambda x: x).not_in(excluded_values)]

        result = TestModel(field=value)
        if value in excluded_values:
            print(f"✗ Value {value} in {excluded_values}: ACCEPTED (should be rejected)")
        else:
            print(f"✓ Value {value} not in {excluded_values}: ACCEPTED (correct)")
    except ValidationError:
        if value in excluded_values:
            print(f"✓ Value {value} in {excluded_values}: REJECTED (correct)")
        else:
            print(f"✗ Value {value} not in {excluded_values}: REJECTED (should be accepted)")

print("\n" + "=" * 60)
print("Testing operator.__not__ behavior directly")
print("=" * 60)

import operator

# Test what the bug report claims
print("Bug report claims:")
print("  '~True returns -2 (truthy)'")
print("  '~False returns -1 (truthy)'")

print("\nActual behavior:")
print(f"  operator.__not__(True) = {operator.__not__(True)} (type: {type(operator.__not__(True))})")
print(f"  operator.__not__(False) = {operator.__not__(False)} (type: {type(operator.__not__(False))})")
print(f"  ~True = {~True} (bitwise NOT)")
print(f"  ~False = {~False} (bitwise NOT)")

# Test the actual logic in the constraint
values = [1, 2, 3]
test_values = [1, 2, 3, 4, 5]

print("\n" + "=" * 60)
print("Testing the actual constraint logic")
print("=" * 60)

for v in test_values:
    contains_result = operator.__contains__(values, v)
    not_result = operator.__not__(contains_result)
    print(f"Value {v}:")
    print(f"  - in {values}: {contains_result}")
    print(f"  - operator.__not__(in): {not_result}")
    print(f"  - Should pass validation: {not_result}")