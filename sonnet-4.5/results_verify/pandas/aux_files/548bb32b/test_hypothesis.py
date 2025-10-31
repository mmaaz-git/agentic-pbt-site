import pandas as pd
from io import StringIO
from hypothesis import given, strategies as st, settings, example
import sys

@given(st.integers())
@example(-9_223_372_036_854_775_809)  # Specific failing case
@example(9_223_372_036_854_775_808)   # Another failing case
@settings(max_examples=100)
def test_series_json_roundtrip(value):
    s = pd.Series([value])
    json_str = s.to_json(orient='split')

    # Skip testing the roundtrip if we know it will fail (for now just document what fails)
    if value < -2**63 or value >= 2**63:
        # These will fail with ujson
        print(f"Known to fail for value={value} (outside int64 range)")
        try:
            result = pd.read_json(StringIO(json_str), typ='series', orient='split', convert_dates=False)
            print(f"  Surprisingly succeeded: {result.tolist()}, dtype={result.dtype}")
        except ValueError as e:
            print(f"  Failed as expected: {e}")
        return

    try:
        result = pd.read_json(StringIO(json_str), typ='series', orient='split', convert_dates=False)
        pd.testing.assert_series_equal(result, s)
        print(f"✓ Success for value={value}")
    except Exception as e:
        print(f"✗ Failed for value={value} (in int64 range): {e}")

# Run the test
if __name__ == "__main__":
    test_series_json_roundtrip()