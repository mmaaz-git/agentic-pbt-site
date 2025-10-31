from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, column
import pandas as pd

@given(data_frames(columns=[
    column('int_col', dtype=int),
    column('float_col', dtype=float),
    column('str_col', dtype=str)
]))
@settings(max_examples=100, deadline=None)
def test_round_trip_mixed_types(df):
    """Round-trip should preserve mixed column types."""
    try:
        interchange_obj = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(interchange_obj)

        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
        print(f"✓ Test passed for shape {df.shape}")
    except UnicodeEncodeError as e:
        # Check if this is a surrogate character issue
        for val in df['str_col']:
            if isinstance(val, str):
                for char in val:
                    if '\ud800' <= char <= '\udfff':
                        print(f"Found surrogate character in string: {repr(val[:20])}")
                        raise e
        raise e

# Run the test
print("Running Hypothesis test...")
try:
    test_round_trip_mixed_types()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed with error: {type(e).__name__}: {e}")
    print("\nThis confirms the bug exists with Hypothesis-generated data.")