import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst
from pandas.core.array_algos import masked_reductions
from pandas._libs import missing as libmissing

@given(
    values=npst.arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=500)
def test_masked_reduction_all_masked(values):
    mask = np.ones(len(values), dtype=bool)

    assert masked_reductions.sum(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.prod(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.min(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.max(values, mask, skipna=True) is libmissing.NA
    assert masked_reductions.mean(values, mask, skipna=True) is libmissing.NA

if __name__ == "__main__":
    test_masked_reduction_all_masked()