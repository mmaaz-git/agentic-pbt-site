from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, column
import pandas as pd

# Create a strategy that specifically generates surrogate characters
surrogate_strategy = st.sampled_from(['\ud800', '\ud801', '\udfff', '\udffe'])

@given(st.data())
def test_with_surrogates(data):
    """Test specifically with surrogate characters."""
    # Generate a string with a surrogate character
    surrogate_char = data.draw(surrogate_strategy)

    df = pd.DataFrame({
        'int_col': [1],
        'float_col': [1.0],
        'str_col': [surrogate_char]
    })

    print(f"Testing with surrogate: {repr(surrogate_char)}")

    try:
        interchange_obj = df.__dataframe__()
        result = pd.api.interchange.from_dataframe(interchange_obj)
        print(f"SUCCESS: Round-trip worked for {repr(surrogate_char)}")
        assert result.shape == df.shape
        assert list(result.columns) == list(df.columns)
    except UnicodeEncodeError as e:
        print(f"FAILED: UnicodeEncodeError for {repr(surrogate_char)}: {e}")
        raise

# Run the test
print("Testing with forced surrogate characters...")
try:
    test_with_surrogates()
    print("\nAll tests passed! (This shouldn't happen if the bug exists)")
except UnicodeEncodeError as e:
    print(f"\nTest failed as expected with UnicodeEncodeError")
    print(f"Error: {e}")
except Exception as e:
    print(f"\nTest failed with unexpected error: {type(e).__name__}: {e}")