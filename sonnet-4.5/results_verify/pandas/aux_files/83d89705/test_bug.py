#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.io.formats.css import CSSResolver
from hypothesis import given, strategies as st

# First test the reproduction case
print("=== Reproducing the Bug ===")
resolver = CSSResolver()

result = resolver.size_to_pt("1e-5pt")
print(f"Input:  1e-5pt")
print(f"Output: {result}")
print(f"Expected value: 1e-05")
print(f"Actual value:   {float(result.rstrip('pt'))}")

result = resolver.size_to_pt("2.5e3px")
print(f"\nInput:  2.5e3px")
print(f"Output: {result}")

# Test with regular notation to confirm it works
print("\n=== Testing regular notation ===")
result = resolver.size_to_pt("0.00001pt")
print(f"Input:  0.00001pt")
print(f"Output: {result}")

result = resolver.size_to_pt("2500px")
print(f"Input:  2500px")
print(f"Output: {result}")

# Now test the property-based test
print("\n=== Running property-based test ===")

@given(
    val=st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(["pt", "px", "em", "rem", "in", "cm", "mm"])
)
def test_size_to_pt_scientific_notation(val, unit):
    input_str = f"{val}{unit}"
    result = resolver.size_to_pt(input_str)
    assert result.endswith("pt"), f"Result {result} should end with 'pt'"
    result_val = float(result.rstrip("pt"))
    assert result_val != 0 or val == 0, f"Non-zero input {input_str} should not produce 0pt"

try:
    test_size_to_pt_scientific_notation()
    print("Property test passed")
except AssertionError as e:
    print(f"Property test failed: {e}")

# Specifically test the failing input mentioned
print("\n=== Testing specific failing input ===")
input_str = f"{1e-5}pt"
print(f"Testing: {input_str}")
result = resolver.size_to_pt(input_str)
print(f"Result: {result}")
print(f"Is result 0pt? {result == '0pt'}")