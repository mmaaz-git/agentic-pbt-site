import pandas as pd
from hypothesis import given, settings, seed
from hypothesis.extra.pandas import column, data_frames

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float)
]))
@settings(max_examples=200)
@seed(42)  # For reproducibility
def test_to_dict_tight_from_dict_tight_roundtrip(df):
    dict_repr = df.to_dict(orient='tight')
    result = pd.DataFrame.from_dict(dict_repr, orient='tight')
    try:
        assert result.equals(df), f"Round-trip with orient='tight' should preserve DataFrame. Original dtypes: {df.dtypes.to_dict()}, Reconstructed dtypes: {result.dtypes.to_dict()}"
    except AssertionError as e:
        # Log the failure for analysis
        print(f"FAILED: {e}")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame empty: {df.empty}")
        if df.empty:
            print("Empty DataFrame detected - this is the bug case")
        raise

# Run the test
if __name__ == "__main__":
    test_to_dict_tight_from_dict_tight_roundtrip()
    print("Test completed")