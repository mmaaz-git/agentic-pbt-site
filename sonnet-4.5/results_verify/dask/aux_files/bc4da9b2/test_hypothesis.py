import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

@settings(max_examples=50)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=-100, max_value=100),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.text(max_size=10)
        ),
        min_size=1,
        max_size=50
    ),
    st.integers(min_value=1, max_value=5)
)
def test_from_pandas_roundtrip(rows, npartitions):
    df_pandas = pd.DataFrame(rows, columns=['a', 'b', 'c'])

    df_dask = dd.from_pandas(df_pandas, npartitions=npartitions)
    result = df_dask.compute()

    pd.testing.assert_frame_equal(df_pandas, result)

if __name__ == "__main__":
    # Run the test
    test_from_pandas_roundtrip()