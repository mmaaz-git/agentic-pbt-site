import sys
import numpy as np

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import assume, given, settings, strategies as st
import pandas.core.array_algos.masked_reductions as masked_reductions
from pandas._libs import missing as libmissing

@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6), min_size=1, max_size=100),
    mask_bits=st.lists(st.booleans(), min_size=1, max_size=100)
)
@settings(max_examples=1000)
def test_masked_reductions_consistency(values, mask_bits):
    assume(len(values) == len(mask_bits))
    assume(all(mask_bits))  # Only test when ALL values are masked

    arr = np.array(values)
    mask = np.array(mask_bits)

    sum_result = masked_reductions.sum(arr, mask, skipna=True)
    min_result = masked_reductions.min(arr, mask, skipna=True)
    mean_result = masked_reductions.mean(arr, mask, skipna=True)

    # The bug report claims these should all be NA
    assert sum_result is libmissing.NA
    assert min_result is libmissing.NA
    assert mean_result is libmissing.NA

print("Running the hypothesis test from bug report...")
print("This test assumes all reduction functions should return NA when all values are masked")
print("=" * 70)

try:
    test_masked_reductions_consistency()
    print("Test passed!")
except AssertionError as e:
    print("Test failed as expected!")
    print("The test fails because sum() returns 0.0 instead of NA")

    # Test the specific failing case mentioned
    print("\n" + "=" * 70)
    print("Testing specific failing case: values=[0.0], mask_bits=[True]")
    values = [0.0]
    mask_bits = [True]
    arr = np.array(values)
    mask = np.array(mask_bits)

    sum_result = masked_reductions.sum(arr, mask, skipna=True)
    print(f"sum([0.0] with all masked) = {sum_result}")
    print(f"Is it NA? {sum_result is libmissing.NA}")
    print(f"Is it 0.0? {sum_result == 0.0}")