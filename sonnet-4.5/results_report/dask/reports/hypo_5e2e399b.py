from hypothesis import given, strategies as st
import pandas as pd
from dask.dataframe.utils import _maybe_sort

@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_maybe_sort_preserves_index_names(data):
    df = pd.DataFrame({'A': data}, index=pd.Index(range(len(data)), name='A'))
    original_name = df.index.names[0]

    result = _maybe_sort(df, check_index=True)

    assert result.index.names[0] == original_name, \
        f"Index name changed from {original_name} to {result.index.names[0]}"

if __name__ == "__main__":
    test_maybe_sort_preserves_index_names()