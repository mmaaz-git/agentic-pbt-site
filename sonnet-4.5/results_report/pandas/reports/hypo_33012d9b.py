from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(["a", "b", "c", None]), min_size=1, max_size=50)
)
@settings(max_examples=100)
def test_categorical_with_nulls(cat_values):
    """Test that categorical columns with nulls round-trip correctly through interchange protocol."""
    df = pd.DataFrame({"cat_col": pd.Categorical(cat_values)})

    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)

    pd.testing.assert_series_equal(
        result_df["cat_col"].reset_index(drop=True),
        df["cat_col"].reset_index(drop=True),
        check_categorical=True
    )


if __name__ == "__main__":
    # Run the test
    test_categorical_with_nulls()