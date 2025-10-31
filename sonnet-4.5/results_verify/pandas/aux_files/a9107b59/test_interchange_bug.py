import pandas as pd
from hypothesis import given, strategies as st, settings, reproduce_failure
from pandas.api.interchange import from_dataframe

@given(
    data=st.lists(st.integers(), min_size=0, max_size=100),
    col_name=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
@settings(max_examples=1000)
def test_round_trip_integer_column(data, col_name):
    df = pd.DataFrame({col_name: data})

    interchange_df = df.__dataframe__()
    df_roundtrip = from_dataframe(interchange_df)

    assert df.equals(df_roundtrip), f"Round-trip failed: {df} != {df_roundtrip}"

# Test with the specific failing input
def test_specific_failure():
    data = [-9_223_372_036_854_775_809]
    col_name = 'A'
    df = pd.DataFrame({col_name: data})

    print(f"Testing with data={data}, col_name='{col_name}'")
    print(f"DataFrame dtype: {df[col_name].dtype}")

    try:
        interchange_df = df.__dataframe__()
        print("Created interchange object successfully")

        df_roundtrip = from_dataframe(interchange_df)
        print("Round-trip succeeded")
        print(f"Result: {df_roundtrip}")
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")

if __name__ == "__main__":
    print("Running specific failure test:")
    print("="*50)
    test_specific_failure()
    print("\n" + "="*50)
    print("Running hypothesis test (will stop on first failure):")
    try:
        test_round_trip_integer_column()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")