import pandas as pd
from pandas.api.interchange import from_dataframe

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
    print(f"Test failed with assertion error")
    print("Expected (original):", pd.Categorical(['a', None]))
    print("Actual (after roundtrip):")

    # Re-run to show what we get
    df = pd.DataFrame({"cat_col": pd.Categorical(['a', None])})
    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)
    print(result_df["cat_col"])

# Also test with multiple values
try:
    test_categorical_with_nulls(['a', 'b', None, 'c', None])
    print("Test passed for multiple values")
except AssertionError as e:
    print(f"Test failed for multiple values")

    # Re-run to show what we get
    df = pd.DataFrame({"cat_col": pd.Categorical(['a', 'b', None, 'c', None])})
    interchange_df = df.__dataframe__()
    result_df = from_dataframe(interchange_df)
    print("Expected:", pd.Categorical(['a', 'b', None, 'c', None]))
    print("Actual:", result_df["cat_col"].values)