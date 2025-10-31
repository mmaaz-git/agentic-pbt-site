"""Test script to reproduce the ndeepmap bug - fixed version"""

# First, let's test the hypothesis property-based test
from hypothesis import given, settings, strategies as st
import dask.utils

@given(st.lists(st.integers(), min_size=2))
@settings(max_examples=100)
def test_ndeepmap_zero_depth_discards_elements(lst):
    inc = lambda x: x + 1
    result = dask.utils.ndeepmap(0, inc, lst)
    expected_if_only_first = inc(lst[0])
    assert result == expected_if_only_first
    # print(f"Test passed for list: {lst}, result: {result}, expected_if_only_first: {expected_if_only_first}")

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_ndeepmap_zero_depth_discards_elements()
    print("Hypothesis test passed - confirming the bug exists")
except Exception as e:
    print(f"Hypothesis test failed with error: {e}")

# Now let's reproduce the specific example from the bug report
print("\n" + "="*50)
print("Reproducing the specific bug example:")
print("="*50)

inc = lambda x: x + 1

result1 = dask.utils.ndeepmap(0, inc, [1])
print(f"ndeepmap(0, inc, [1]) = {result1}")

result2 = dask.utils.ndeepmap(0, inc, [1, 2, 3, 4, 5])
print(f"ndeepmap(0, inc, [1, 2, 3, 4, 5]) = {result2}")

print(f"\nChecking if both results are equal: {result1 == result2}")
assert result1 == result2
print("Assertion passed - bug confirmed: both results are equal despite different inputs")

# Let's also test with more examples to understand the behavior
print("\n" + "="*50)
print("Additional tests to understand the behavior:")
print("="*50)

# Test with different depths
print("\nTesting with n=0:")
print(f"ndeepmap(0, inc, [10, 20, 30]) = {dask.utils.ndeepmap(0, inc, [10, 20, 30])}")
print(f"ndeepmap(0, inc, [100]) = {dask.utils.ndeepmap(0, inc, [100])}")

print("\nTesting with n=1:")
print(f"ndeepmap(1, inc, [10, 20, 30]) = {dask.utils.ndeepmap(1, inc, [10, 20, 30])}")

print("\nTesting with n=2:")
print(f"ndeepmap(2, lambda x: x+1, [[[1]], [[2]], [[3]]]) = {dask.utils.ndeepmap(2, lambda x: x+1, [[[1]], [[2]], [[3]]])}")

print("\nTesting with negative n:")
print(f"ndeepmap(-1, inc, [10, 20, 30]) = {dask.utils.ndeepmap(-1, inc, [10, 20, 30])}")

# Let's check what happens with empty lists
print("\nTesting with empty list:")
try:
    result = dask.utils.ndeepmap(0, inc, [])
    print(f"ndeepmap(0, inc, []) = {result}")
except Exception as e:
    print(f"ndeepmap(0, inc, []) raised error: {e}")

# Let's test when n=0 with non-list types
print("\nTesting with n=0 and non-list types:")
print(f"ndeepmap(0, inc, 5) = {dask.utils.ndeepmap(0, inc, 5)}")
print(f"ndeepmap(0, inc, (1, 2, 3)) = {dask.utils.ndeepmap(0, inc, (1, 2, 3))}")