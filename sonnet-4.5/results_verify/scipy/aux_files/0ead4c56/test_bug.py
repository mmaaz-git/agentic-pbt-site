#!/usr/bin/env python3
import numpy as np
from scipy.spatial.distance import dice
from hypothesis import given, strategies as st, settings

# First let's reproduce the exact case from the bug report
print("Testing the exact failing case from bug report:")
u = np.array([False, False])
v = np.array([False, False])
result = dice(u, v)
print(f"dice([False, False], [False, False]) = {result}")
print(f"Expected: 0.0")
print(f"Is NaN: {np.isnan(result)}")
print()

# Test with another all-False case
print("Testing with larger all-False arrays:")
u = np.array([False, False, False, False])
v = np.array([False, False, False, False])
result = dice(u, v)
print(f"dice([F,F,F,F], [F,F,F,F]) = {result}")
print(f"Is NaN: {np.isnan(result)}")
print()

# Test that identical arrays normally give 0
print("Testing with identical arrays containing True values:")
u = np.array([True, False, True])
v = np.array([True, False, True])
result = dice(u, v)
print(f"dice([T,F,T], [T,F,T]) = {result}")
print(f"Expected: 0.0")
print()

# Let's test the formula manually for all-False case
print("Manual calculation for all-False case:")
u = np.array([False, False])
v = np.array([False, False])
c_TT = np.sum(u & v)  # True & True
c_TF = np.sum(u & ~v)  # True & False
c_FT = np.sum(~u & v)  # False & True
print(f"c_TT (T&T) = {c_TT}")
print(f"c_TF (T&F) = {c_TF}")
print(f"c_FT (F&T) = {c_FT}")
denominator = 2*c_TT + c_TF + c_FT
numerator = c_TF + c_FT
print(f"Numerator (c_TF + c_FT) = {numerator}")
print(f"Denominator (2*c_TT + c_TF + c_FT) = {denominator}")
if denominator == 0:
    print("Division by zero detected! This causes NaN")
print()

# Run the hypothesis test
print("Running hypothesis test:")
@given(
    st.integers(min_value=2, max_value=30),
    st.integers(min_value=0, max_value=1000)
)
@settings(max_examples=100)  # Reduced for quicker testing
def test_dice_identical_is_zero(n, seed):
    np.random.seed(seed)
    u = np.random.randint(0, 2, n, dtype=bool)

    d = dice(u, u)

    assert not np.isnan(d), \
        f"Dice distance should not return NaN for u={u}"
    assert np.allclose(d, 0.0, rtol=1e-10, atol=1e-10), \
        f"Dice distance of identical vectors should be 0, got {d} for u={u}"

try:
    test_dice_identical_is_zero()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")