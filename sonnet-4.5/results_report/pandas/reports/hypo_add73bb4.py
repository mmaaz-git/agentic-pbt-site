import pandas.arrays as pa
import numpy as np
from hypothesis import given, strategies as st, settings
import sys

# Set a lower recursion limit to catch the bug faster
sys.setrecursionlimit(100)

def sparse_array_strategy(min_size=0, max_size=50):
    @st.composite
    def _strat(draw):
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        fill_value = draw(st.sampled_from([0, 0.0, -1, 1]))
        values = draw(st.lists(
            st.sampled_from([fill_value, 1, 2, 3, -1, 10]),
            min_size=size, max_size=size
        ))
        return pa.SparseArray(values, fill_value=fill_value)
    return _strat()


@given(sparse_array_strategy(min_size=1, max_size=20))
@settings(max_examples=100)
def test_sparsearray_cumsum_doesnt_crash(arr):
    result = arr.cumsum()
    assert isinstance(result, pa.SparseArray)

if __name__ == "__main__":
    test_sparsearray_cumsum_doesnt_crash()