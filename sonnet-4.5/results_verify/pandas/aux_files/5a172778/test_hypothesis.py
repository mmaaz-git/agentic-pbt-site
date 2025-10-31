import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import column, data_frames, range_indexes


@given(
    x=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=1e6), min_size=20, max_size=100),
    q=st.integers(2, 10),
)
@settings(max_examples=300)
def test_qcut_equal_sized_bins(x, q):
    s = pd.Series(x)
    assume(len(s.unique()) >= q)
    result = pd.qcut(s, q=q, duplicates="drop")
    counts = result.value_counts()
    max_count = counts.max()
    min_count = counts.min()
    assert max_count - min_count <= 2

# Run the test
if __name__ == "__main__":
    test_qcut_equal_sized_bins()
    print("Test completed - no issues found in automatic testing")