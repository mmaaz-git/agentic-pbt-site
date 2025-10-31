from hypothesis import given, strategies as st, settings
import pandas as pd
from pandas.api.interchange import from_dataframe

@given(
    st.lists(
        st.one_of(st.integers(min_value=0, max_value=100), st.none()),
        min_size=1,
        max_size=100
    )
)
@settings(max_examples=1000)
def test_roundtrip_nullable_integers(data):
    df = pd.DataFrame({"col": pd.array(data, dtype="Int64")})

    interchange_obj = df.__dataframe__()
    result_df = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result_df, df)

# Test with specific failing example
print("Testing specific failing case: [None]")
try:
    test_roundtrip_nullable_integers([None])
    print("Test passed for [None]")
except AssertionError as e:
    print(f"Test failed for [None]: {e}")

print("\nTesting specific failing case: [1, None, 3]")
try:
    test_roundtrip_nullable_integers([1, None, 3])
    print("Test passed for [1, None, 3]")
except AssertionError as e:
    print(f"Test failed for [1, None, 3]: {e}")

# Run hypothesis testing
print("\nRunning hypothesis tests...")
try:
    test_roundtrip_nullable_integers()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")