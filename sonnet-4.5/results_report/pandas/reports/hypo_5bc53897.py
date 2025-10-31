import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray
from hypothesis import given, strategies as st


@given(st.lists(st.one_of(st.integers(), st.none()), min_size=1))
def test_fillna_preserves_non_na(values):
    arr = ArrowExtensionArray(pa.array(values))
    filled = arr.fillna(value=999)

    for i in range(len(arr)):
        if not pd.isna(arr[i]):
            assert arr[i] == filled[i]


if __name__ == "__main__":
    test_fillna_preserves_non_na()