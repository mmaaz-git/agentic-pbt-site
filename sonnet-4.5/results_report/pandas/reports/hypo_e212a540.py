import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings, example
from pandas.arrays import SparseArray


@given(st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=50))
@example([0])  # Minimal example that fails
@settings(max_examples=1)
def test_cumsum_matches_dense(data):
    arr = SparseArray(data)
    dense = arr.to_dense()

    sparse_cumsum = arr.cumsum()
    dense_cumsum = np.cumsum(dense)

    assert np.array_equal(sparse_cumsum.to_dense(), dense_cumsum)


if __name__ == "__main__":
    test_cumsum_matches_dense()