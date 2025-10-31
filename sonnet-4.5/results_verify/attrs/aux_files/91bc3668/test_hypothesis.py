import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings


@given(
    st.lists(
        st.text(min_size=1, max_size=10),
        min_size=5,
        max_size=30
    )
)
@settings(max_examples=50)
def test_str_upper_matches_pandas(strings):
    pdf = pd.DataFrame({'text': strings})
    ddf = dd.from_pandas(pdf, npartitions=2)

    pandas_result = pdf['text'].str.upper()
    dask_result = ddf['text'].str.upper().compute()

    for i in range(len(strings)):
        assert pandas_result.iloc[i] == dask_result.iloc[i], \
            f"str.upper() mismatch for '{strings[i]}': pandas='{pandas_result.iloc[i]}', dask='{dask_result.iloc[i]}'"

if __name__ == "__main__":
    # Run the test
    test_str_upper_matches_pandas()