import pandas as pd
import pandas.api.interchange as interchange
from hypothesis import given, strategies as st
from hypothesis.extra.pandas import data_frames, columns, column

# First, let's reproduce the basic bug
print("=== Testing basic surrogate character bug ===")
try:
    df = pd.DataFrame({
        'A': [0],
        'B': [0.0],
        'C': ['\ud800']  # Surrogate character
    })

    print(f"Original DataFrame created successfully:")
    print(df)
    print(f"String column contains: {repr(df['C'][0])}")

    # Try to convert via interchange protocol
    interchange_obj = df.__dataframe__()
    print("__dataframe__() succeeded")

    result = interchange.from_dataframe(interchange_obj)
    print("from_dataframe() succeeded")
    print(f"Result DataFrame: {result}")

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

# Test with hypothesis
print("\n=== Running Hypothesis test ===")

@given(data_frames([
    column('A', dtype=int),
    column('B', dtype=float),
    column('C', dtype=str),
]))
def test_roundtrip_preserves_shape(df):
    try:
        interchange_obj = df.__dataframe__()
        result = interchange.from_dataframe(interchange_obj)
        assert result.shape == df.shape
        return True
    except UnicodeEncodeError:
        # Check if df contains surrogates
        for col in df.columns:
            if df[col].dtype == 'object':
                for val in df[col]:
                    if isinstance(val, str):
                        for char in val:
                            if 0xD800 <= ord(char) <= 0xDFFF:
                                print(f"Found surrogate in generated data: {repr(val)}")
                                raise
        raise
    except Exception as e:
        print(f"Other error in hypothesis test: {type(e).__name__}: {e}")
        raise

# Run a limited test
try:
    test_roundtrip_preserves_shape()
    print("Hypothesis test completed without finding the bug")
except Exception as e:
    print(f"Hypothesis test found issue: {e}")

# Test other edge cases with surrogates
print("\n=== Testing other surrogate cases ===")
test_cases = [
    '\ud800',  # High surrogate alone
    '\udc00',  # Low surrogate alone
    '\ud800\udc00',  # Valid surrogate pair (should work in UTF-16)
    'normal\ud800text',  # Surrogate in middle
]

for test_str in test_cases:
    try:
        df = pd.DataFrame({'col': [test_str]})
        interchange_obj = df.__dataframe__()
        result = interchange.from_dataframe(interchange_obj)
        print(f"✓ Handled {repr(test_str)}")
    except UnicodeEncodeError as e:
        print(f"✗ Failed on {repr(test_str)}: UnicodeEncodeError")
    except Exception as e:
        print(f"✗ Failed on {repr(test_str)}: {type(e).__name__}: {e}")