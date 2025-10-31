"""Test to reproduce the _apply_constraint bug"""

import annotated_types
import pytest
from hypothesis import given, strategies as st, settings
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint


# First, run the property-based test
@settings(max_examples=500)
@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_constraint_schema_structure(value):
    """
    Property: Constraint application should be consistent across similar constraint types.

    Gt (correctly) applies constraint once in schema only.
    Ge (bug) applies constraint in schema AND wraps with validator function.
    """
    int_schema = cs.int_schema()

    gt_schema = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    ge_schema = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))

    assert gt_schema['type'] == 'int'

    if ge_schema['type'] != gt_schema['type']:
        pytest.fail(f"Ge produces {ge_schema['type']} but Gt produces {gt_schema['type']}")


# Now run the manual reproduction
def manual_reproduction():
    """Manually reproduce the bug"""
    int_schema = cs.int_schema()

    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(10))
    print(f"Gt schema: {gt_result}")

    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(10))
    print(f"Ge schema: {ge_result}")

    # Check other affected constraints
    lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(10))
    print(f"Lt schema: {lt_result}")

    le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(10))
    print(f"Le schema: {le_result}")

    # Test MultipleOf
    multiple_of_result = _apply_constraint(int_schema.copy(), annotated_types.MultipleOf(5))
    print(f"MultipleOf schema: {multiple_of_result}")

    # Test Len on string
    str_schema = cs.str_schema()
    len_result = _apply_constraint(str_schema.copy(), annotated_types.Len(min_length=5, max_length=10))
    print(f"Len schema: {len_result}")


if __name__ == "__main__":
    print("Running manual reproduction:")
    print("-" * 50)
    manual_reproduction()
    print("\n" + "=" * 50)
    print("\nRunning hypothesis test:")
    try:
        # Run the test directly without hypothesis wrapper
        int_schema = cs.int_schema()
        gt_schema = _apply_constraint(int_schema.copy(), annotated_types.Gt(0))
        ge_schema = _apply_constraint(int_schema.copy(), annotated_types.Ge(0))

        print(f"Gt schema type: {gt_schema['type']}")
        print(f"Ge schema type: {ge_schema['type']}")

        if ge_schema['type'] != gt_schema['type']:
            print(f"Test failed as expected: Ge produces {ge_schema['type']} but Gt produces {gt_schema['type']}")
        else:
            print("Test passed (this shouldn't happen if the bug exists)")
    except Exception as e:
        print(f"Error during test: {e}")