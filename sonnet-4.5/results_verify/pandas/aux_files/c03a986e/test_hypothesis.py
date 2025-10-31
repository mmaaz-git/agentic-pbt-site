import pandas as pd
from hypothesis import given, settings, reproduce_failure
from hypothesis.extra.pandas import column, data_frames

@given(data_frames([
    column('a', dtype=int),
    column('b', dtype=float)
]))
@settings(max_examples=200)
def test_to_dict_index_from_dict_index_roundtrip(df):
    dict_repr = df.to_dict(orient='index')
    result = pd.DataFrame.from_dict(dict_repr, orient='index')

    # Debug output for failures
    if not result.equals(df):
        print("\n*** FAILURE FOUND ***")
        print(f"Original df shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Dict repr: {dict_repr}")
        print(f"Result shape: {result.shape}")
        print(f"Result columns: {result.columns.tolist()}")

    assert result.equals(df), f"Round-trip with orient='index' should preserve DataFrame. Original: {df.shape}, Result: {result.shape}"

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_to_dict_index_from_dict_index_roundtrip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")