#!/usr/bin/env python3

import numpy.rec
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_array_handles_all_list_sizes(lst):
    result = numpy.rec.array(lst)
    assert isinstance(result, numpy.rec.recarray)

# Run the hypothesis test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_array_handles_all_list_sizes()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")