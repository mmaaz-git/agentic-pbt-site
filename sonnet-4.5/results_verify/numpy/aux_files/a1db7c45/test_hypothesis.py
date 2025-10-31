#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, settings, example

@given(st.lists(st.text(min_size=1), min_size=1).map(lambda x: np.array(x, dtype=str)),
       st.text(min_size=1, max_size=5),
       st.integers(min_value=0, max_value=20),
       st.one_of(st.integers(min_value=0, max_value=20), st.none()))
@example(np.array(['abc'], dtype=str), '\x00', 0, None)  # The specific failing case
@settings(max_examples=100)  # Reduced for faster testing
def test_find_with_bounds(arr, sub, start, end):
    result = nps.find(arr, sub, start, end)
    for i in range(len(arr)):
        expected = arr[i].find(sub, start, end)
        if result[i] != expected:
            print(f"\nFAILURE FOUND:")
            print(f"  String: {repr(arr[i])}")
            print(f"  Substring: {repr(sub)}")
            print(f"  Start: {start}, End: {end}")
            print(f"  Expected (Python): {expected}")
            print(f"  Got (NumPy): {result[i]}")
            assert False, f"Mismatch for find({repr(arr[i])}, {repr(sub)}, {start}, {end})"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_find_with_bounds()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")