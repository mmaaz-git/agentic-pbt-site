from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

@settings(max_examples=500)
@given(
    st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    st.lists(st.integers(min_value=0, max_value=99), min_size=0, max_size=20)
)
def test_take_multiple_indices(data, indices):
    assume(all(idx < len(data) for idx in indices))
    arr = ArrowExtensionArray._from_sequence(data, dtype=pd.ArrowDtype(pa.int64()))
    result = arr.take(indices)
    expected = [data[idx] for idx in indices]
    assert result.tolist() == expected

if __name__ == "__main__":
    # Run the test
    test_take_multiple_indices()