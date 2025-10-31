#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

from scipy.io.arff._arffread import NumericAttribute
import numpy as np

print("Testing _basic_stats with single-element array:")
print("-" * 50)

attr = NumericAttribute("test")
data = np.array([5.0])

min_val, max_val, mean_val, std_val = attr._basic_stats(data)

print(f"Input data: {data}")
print(f"min: {min_val}")
print(f"max: {max_val}")
print(f"mean: {mean_val}")
print(f"std: {std_val}")
print(f"std is infinite: {np.isinf(std_val)}")
print(f"std is nan: {np.isnan(std_val)}")

print("\n" + "="*50)
print("Testing with 2-element array for comparison:")
data2 = np.array([5.0, 10.0])
min_val2, max_val2, mean_val2, std_val2 = attr._basic_stats(data2)
print(f"Input data: {data2}")
print(f"min: {min_val2}")
print(f"max: {max_val2}")
print(f"mean: {mean_val2}")
print(f"std: {std_val2}")

print("\n" + "="*50)
print("Testing mathematical calculations:")
print(f"For single element [5.0]:")
print(f"  nbfac = 1 / (1 - 1) = 1 / 0 = {1.0 / (1 - 1) if 1 != 1 else 'inf (division by zero)'}")
print(f"  np.std([5.0]) = {np.std(np.array([5.0]))}")
print(f"  0 * inf = {0 * float('inf')}")

print("\n" + "="*50)
print("Testing the hypothesis test:")
from hypothesis import given, strategies as st

@given(value=st.floats(allow_nan=False, allow_infinity=False,
                       min_value=-1e6, max_value=1e6))
def test_basic_stats_single_element_should_be_finite(value):
    """_basic_stats should produce finite statistics for single-element arrays"""
    attr = NumericAttribute("test")
    data = np.array([value])

    min_val, max_val, mean_val, std_val = attr._basic_stats(data)

    assert np.isfinite(std_val), \
        f"Standard deviation should be finite for single element, got {std_val}"

# Run the property test
print("Running property-based test...")
try:
    test_basic_stats_single_element_should_be_finite()
    print("Property test passed!")
except AssertionError as e:
    print(f"Property test FAILED: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")