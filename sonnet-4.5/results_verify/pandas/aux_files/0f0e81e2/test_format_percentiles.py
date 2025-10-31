"""Test reproduction for pandas.core.methods.describe.format_percentiles bug"""

from pandas.core.methods.describe import format_percentiles
import numpy as np

# Test 1: Basic failing case
print("Test 1: Basic failing case")
percentiles = [0.0, 2.225073858507203e-309]
result = format_percentiles(percentiles)
print(f"Input:  {percentiles}")
print(f"Output: {result}")
print(f"Issue: Two different values both formatted as '0%'\n")

# Test 2: Produces 'inf%' outputs
print("Test 2: Produces 'inf%' outputs")
percentiles2 = [0.9374708737308688, 0.5, 0.22976323147538885, 0.8834939856757813, 2.225073858507203e-309]
result2 = format_percentiles(percentiles2)
print(f"Input:  {percentiles2}")
print(f"Output: {result2}")
print(f"Issue: Invalid 'inf%' strings in output\n")

# Test 3: Produces 'nan%' outputs
print("Test 3: Produces 'nan%' outputs")
percentiles3 = [5e-324, 9.889517165452854e-55, 0.027853332816048855, 0.5]
result3 = format_percentiles(percentiles3)
print(f"Input:  {percentiles3}")
print(f"Output: {result3}")
print(f"Issue: Invalid 'nan%' strings in output\n")

# Run the hypothesis test
print("Running Hypothesis property test...")
from hypothesis import given, strategies as st, assume

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20))
def test_format_percentiles_uniqueness_preservation(percentiles):
    unique_input = np.unique(percentiles)
    assume(len(unique_input) >= 2)

    result = format_percentiles(percentiles)
    unique_output = set(result)

    assert len(unique_input) == len(unique_output), \
        f"Uniqueness not preserved: {len(unique_input)} unique inputs -> {len(unique_output)} unique outputs"

# Run test with the specific failing case
try:
    test_format_percentiles_uniqueness_preservation([0.0, 2.225073858507203e-309])
    print("Hypothesis test passed (should not happen)")
except AssertionError as e:
    print(f"Hypothesis test failed as expected: {e}")