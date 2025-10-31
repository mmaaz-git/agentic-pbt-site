#!/usr/bin/env python3
"""Test to reproduce the reported bug in pydantic.experimental.pipeline._apply_constraint"""

from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import _apply_constraint
import annotated_types
from pydantic_core import core_schema as cs


print("Testing _apply_constraint behavior...")
print("=" * 60)

# First, let's run the basic reproduction test
int_schema = cs.int_schema()

gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))

print(f"Gt result type: {gt_result['type']}")
print(f"Gt result: {gt_result}")
print()
print(f"Ge result type: {ge_result['type']}")
print(f"Ge result: {ge_result}")
print()

# Now test for the other constraints mentioned
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(5))
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(5))

print(f"Lt result type: {lt_result['type']}")
print(f"Lt result: {lt_result}")
print()
print(f"Le result type: {le_result['type']}")
print(f"Le result: {le_result}")
print()

# Test Len constraint
str_schema = cs.str_schema()
len_result = _apply_constraint(str_schema.copy(), annotated_types.Len(5, 10))
print(f"Len result type: {len_result['type']}")
print(f"Len result: {len_result}")
print()

# Test MultipleOf constraint
multiple_of_result = _apply_constraint(int_schema.copy(), annotated_types.MultipleOf(3))
print(f"MultipleOf result type: {multiple_of_result['type']}")
print(f"MultipleOf result: {multiple_of_result}")
print()

# Now run the hypothesis test
print("=" * 60)
print("Running hypothesis test...")

@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_schema_consistency_with_gt(value):
    int_schema = cs.int_schema()

    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))

    assert gt_result['type'] == 'int', "Gt should return int schema"
    assert ge_result['type'] == 'int', f"Ge should return int schema like Gt, got '{ge_result['type']}'"

try:
    test_ge_schema_consistency_with_gt()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")

print("=" * 60)
print("\nAssertion test from the bug report:")
try:
    assert gt_result['type'] == 'int'
    print("✓ gt_result['type'] == 'int'")
except AssertionError:
    print("✗ gt_result['type'] == 'int'")

try:
    assert ge_result['type'] == 'int'
    print("✓ ge_result['type'] == 'int'")
except AssertionError:
    print("✗ ge_result['type'] == 'int'")