#!/usr/bin/env python3
"""Reproduce the typo bug in pandas.core.sample.preprocess_weights"""

from hypothesis import given, strategies as st
import pytest
import numpy as np
from pandas import DataFrame
from pandas.core.sample import preprocess_weights

# First, test with the Hypothesis test from the bug report
@given(
    n_rows=st.integers(min_value=1, max_value=100),
    axis=st.integers(min_value=0, max_value=1)
)
def test_preprocess_weights_negative_error_message(n_rows, axis):
    df = DataFrame(np.random.randn(n_rows, 3))
    shape = n_rows if axis == 0 else 3
    weights = np.ones(shape, dtype=np.float64)
    weights[0] = -1.0

    with pytest.raises(ValueError) as exc_info:
        preprocess_weights(df, weights, axis)

    error_msg = str(exc_info.value)
    # Bug: message says "many" instead of "may"
    assert "weight vector many not include negative values" == error_msg

# Run the hypothesis test
print("Running Hypothesis test...")
try:
    test_preprocess_weights_negative_error_message()
    print("Hypothesis test passed - confirms the typo exists")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now test with the direct example from the bug report
print("\nRunning direct reproduction example...")
df = DataFrame(np.random.randn(5, 3))
weights = np.array([1.0, 2.0, -1.0, 3.0, 4.0])

try:
    preprocess_weights(df, weights, axis=0)
except ValueError as e:
    print(f"Error message captured: '{e}'")
    print(f"Contains typo 'many'?: {'many' in str(e)}")
    print(f"Should be 'may' instead?: Yes")

# Additional test - verify the exact error message
print("\nVerifying exact error message...")
df = DataFrame([[1, 2], [3, 4]])
weights = np.array([-1.0, 1.0])

try:
    preprocess_weights(df, weights, axis=0)
except ValueError as e:
    actual_msg = str(e)
    expected_wrong = "weight vector many not include negative values"
    expected_correct = "weight vector may not include negative values"

    print(f"Actual error message: '{actual_msg}'")
    print(f"Matches wrong version (with typo): {actual_msg == expected_wrong}")
    print(f"Matches correct version (without typo): {actual_msg == expected_correct}")