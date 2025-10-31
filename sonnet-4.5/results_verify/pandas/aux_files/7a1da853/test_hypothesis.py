#!/usr/bin/env python3

from io import StringIO
import pandas as pd
from hypothesis import given, settings, strategies as st, assume

@given(
    st.lists(
        st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=5),
        min_size=1,
        max_size=10
    )
)
@settings(max_examples=500)
def test_dataframe_roundtrip_columns(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    df = pd.DataFrame(data)
    assume(df.columns.is_unique)

    json_str = df.to_json(orient='columns')
    df_recovered = pd.read_json(StringIO(json_str), orient='columns')

    pd.testing.assert_frame_equal(df, df_recovered, check_dtype=False)

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_dataframe_roundtrip_columns()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed with error: {e}")

    # Test the specific failing input
    print("\nTesting specific failing input: data=[[1.7976931345e+308]]")
    try:
        test_dataframe_roundtrip_columns([[1.7976931345e+308]])
        print("Specific test passed!")
    except AssertionError as e:
        print(f"Specific test failed as expected: {e}")