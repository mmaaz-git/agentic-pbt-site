from hypothesis import given, strategies as st, settings
import io
import pandas as pd
from pandas import Series

@given(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                min_size=1, max_size=50))
@settings(max_examples=200)
def test_series_roundtrip_index(data):
    s = Series(data)
    json_str = s.to_json(orient='index')
    result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
    pd.testing.assert_series_equal(s, result)

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with data=[0.0]")
    try:
        # Directly test the failing case
        s = Series([0.0])
        json_str = s.to_json(orient='index')
        result = pd.read_json(io.StringIO(json_str), typ='series', orient='index')
        pd.testing.assert_series_equal(s, result)
        print("Test passed")
    except AssertionError as e:
        print(f"Test failed: {e}")

    print("\nRunning Hypothesis tests...")
    from hypothesis import find
    try:
        # Find a failing example
        failing_data = find(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                                     min_size=1, max_size=50),
                           lambda data: (
                               s := Series(data),
                               json_str := s.to_json(orient='index'),
                               result := pd.read_json(io.StringIO(json_str), typ='series', orient='index'),
                               s.dtype != result.dtype
                           )[-1])
        print(f"Found failing example: {failing_data}")
    except:
        print("Could not find a failing example automatically")