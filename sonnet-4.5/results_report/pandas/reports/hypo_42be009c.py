import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.extensions import take


@given(
    values=st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1000, max_value=1000), min_size=2, max_size=20),
    n_valid=st.integers(min_value=0, max_value=5),
    n_missing=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=300)
def test_index_allow_fill_none_should_fill_with_na(values, n_valid, n_missing):
    index = pd.Index(values, dtype='float64')
    arr = np.array(values)

    valid_idx = list(range(min(n_valid, len(values))))
    missing_idx = [-1] * n_missing
    indices = valid_idx + missing_idx

    index_result = take(index, indices, allow_fill=True, fill_value=None)
    array_result = take(arr, indices, allow_fill=True, fill_value=None)

    for i in range(len(indices)):
        if indices[i] == -1:
            assert pd.isna(array_result[i]), "Array should have NaN for -1"
            assert pd.isna(index_result[i]), f"Index should have NaN for -1, got {index_result[i]}"


if __name__ == "__main__":
    test_index_allow_fill_none_should_fill_with_na()