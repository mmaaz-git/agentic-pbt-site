from hypothesis import given, settings
from hypothesis.extra import numpy as npst
from hypothesis import strategies as st
import numpy as np
import pandas.core.array_algos.take as take_module

@given(
    arr=npst.arrays(
        dtype=st.sampled_from([np.int32, np.int64]),
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=5, max_side=20),
    ),
    indexer_size=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_take_1d_mask_consistency(arr, indexer_size):
    arr_len = len(arr)
    indexer = np.random.randint(0, arr_len, size=indexer_size, dtype=np.intp)

    # Create an all-False mask (no masking needed)
    mask_all_false = np.zeros(indexer_size, dtype=bool)

    # Test with explicit all-False mask
    result1 = take_module.take_1d(arr, indexer, fill_value=np.nan, allow_fill=True, mask=mask_all_false)

    # Test with mask=None (no -1 in indexer, so no masking needed)
    result2 = take_module.take_1d(arr, indexer, fill_value=np.nan, allow_fill=True, mask=None)

    # Both should have the same dtype since no masking actually occurs
    assert result1.dtype == result2.dtype, f"Inconsistent dtypes: {result1.dtype} vs {result2.dtype}"

if __name__ == "__main__":
    test_take_1d_mask_consistency()
    print("Test passed!")