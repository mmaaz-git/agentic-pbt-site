import pandas as pd
import pandas.arrays as pa
from hypothesis import given, settings, strategies as st
import traceback

print("=" * 60)
print("Testing pandas.arrays.IntervalArray.overlaps")
print("=" * 60)

# Test 1: Simple reproduction case from the bug report
print("\nTest 1: Simple case with IntervalArray")
try:
    arr = pa.IntervalArray.from_tuples([(0, 1), (2, 3)])
    print(f"Created IntervalArray: {arr}")
    result = arr.overlaps(arr)
    print(f"Result of arr.overlaps(arr): {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: The specific failing input from hypothesis test
print("\nTest 2: Specific failing input from hypothesis")
try:
    arr = pa.IntervalArray.from_tuples([(0.0, 0.0)])
    print(f"Created IntervalArray: {arr}")
    result = arr.overlaps(arr)
    print(f"Result of arr.overlaps(arr): {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 3: Check if it works with a scalar Interval
print("\nTest 3: Using scalar Interval instead")
try:
    arr = pa.IntervalArray.from_tuples([(0, 1), (2, 3)])
    print(f"Created IntervalArray: {arr}")
    interval = pd.Interval(0.5, 1.5)
    print(f"Created Interval: {interval}")
    result = arr.overlaps(interval)
    print(f"Result of arr.overlaps(interval): {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 4: Run the hypothesis test
print("\nTest 4: Running hypothesis test")
def interval_strategy(min_val=-100, max_val=100):
    return st.tuples(
        st.floats(allow_nan=False, allow_infinity=False, min_value=min_val, max_value=max_val),
        st.floats(allow_nan=False, allow_infinity=False, min_value=min_val, max_value=max_val)
    ).filter(lambda t: t[0] <= t[1])

@given(st.lists(interval_strategy(), min_size=1, max_size=20))
@settings(max_examples=5)
def test_intervalarray_overlaps_reflexive(intervals):
    arr = pa.IntervalArray.from_tuples(intervals)
    result = arr.overlaps(arr)
    for i, (interval, overlaps_self) in enumerate(zip(arr, result)):
        if pd.notna(interval):
            assert overlaps_self

try:
    test_intervalarray_overlaps_reflexive()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    traceback.print_exc()

# Test 5: Check what the documentation says the parameter should be
print("\nTest 5: Checking method signature and docstring")
print(f"Method signature: {pa.IntervalArray.overlaps.__name__}")
print(f"Docstring preview:")
print(pa.IntervalArray.overlaps.__doc__[:500] if pa.IntervalArray.overlaps.__doc__ else "No docstring")