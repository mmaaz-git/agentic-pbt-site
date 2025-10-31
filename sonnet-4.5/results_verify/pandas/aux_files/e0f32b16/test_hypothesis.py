import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')
from hypothesis import given, strategies as st, settings
from pandas.core.array_algos.masked_reductions import sum as masked_sum
from pandas._libs import missing as libmissing


@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100)
)
@settings(max_examples=500)
def test_masked_sum_all_masked_returns_na(values):
    arr = np.array(values, dtype=np.int64)
    mask = np.ones(len(arr), dtype=bool)

    result = masked_sum(arr, mask, skipna=True)
    assert result is libmissing.NA, f"Expected NA but got {result} for values {values}"

# Run the test
try:
    test_masked_sum_all_masked_returns_na()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")