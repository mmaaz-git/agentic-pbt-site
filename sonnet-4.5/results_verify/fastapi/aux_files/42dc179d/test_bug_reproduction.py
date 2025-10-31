#!/usr/bin/env python3
"""Test script to reproduce the ServerVariable bug"""

# First, let's run the simple reproduction case
print("=" * 60)
print("Simple Reproduction Test")
print("=" * 60)

from fastapi.openapi.models import ServerVariable

try:
    sv = ServerVariable(
        enum=["production", "staging", "development"],
        default="invalid_environment"
    )

    print(f"enum: {sv.enum}")
    print(f"default: {sv.default}")
    print(f"Is default in enum? {sv.default in sv.enum}")
    print("\nResult: ServerVariable created successfully with invalid default!")
    print("This should have raised a ValidationError but didn't.")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Property-Based Test")
print("=" * 60)

# Now run the hypothesis-based test
from hypothesis import given, strategies as st, settings

@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10),
    st.text(min_size=1, max_size=20)
)
@settings(max_examples=200)
def test_server_variable_default_not_validated(enum_values, default):
    enum_values = list(set(enum_values))
    if len(enum_values) < 2:
        return

    if default in enum_values:
        return

    sv = ServerVariable(
        enum=enum_values,
        default=default
    )

    assert sv.default == default
    assert sv.default not in sv.enum
    print(f"Test passed with enum={enum_values[:3]}..., default={default}")

# Run the hypothesis test
try:
    test_server_variable_default_not_validated()
    print("\nHypothesis test completed - no errors raised for invalid defaults")
except Exception as e:
    print(f"Hypothesis test failed with: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Specific Failing Example from Bug Report")
print("=" * 60)

# Test the specific failing input mentioned
try:
    sv = ServerVariable(
        enum=["production", "staging"],
        default="invalid"
    )
    print(f"Created ServerVariable with enum={sv.enum}, default={sv.default}")
    print(f"Is default in enum? {sv.default in sv.enum}")
    print("Result: No validation error raised!")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}: {e}")