from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe

@given(
    st.lists(st.sampled_from(["a", "b", "c", None]), min_size=1, max_size=50)
)
def test_categorical_with_nulls(cat_values):
    df = pd.DataFrame({"cat_col": pd.Categorical(cat_values)})

    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)

    pd.testing.assert_series_equal(
        result_df["cat_col"].reset_index(drop=True),
        df["cat_col"].reset_index(drop=True)
    )

# Run a specific test case that should fail
try:
    test_categorical_with_nulls(['a', None])
    print("Test passed")
except AssertionError as e:
    print(f"Test failed with assertion error: {e}")

# Also test with multiple values
try:
    test_categorical_with_nulls(['a', 'b', None, 'c', None])
    print("Test passed for multiple values")
except AssertionError as e:
    print(f"Test failed for multiple values: {e}")