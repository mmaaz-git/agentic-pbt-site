from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import math

@given(st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=50),
       st.integers(2, 10))
@settings(max_examples=100)
def test_cut_coverage(data, n_bins):
    assume(len(set(data)) > 1)
    try:
        result = pd.cut(data, bins=n_bins)
        assert len(result) == len(data)
        print(f"✓ Test passed with data={data[:3]}{'...' if len(data) > 3 else ''}, n_bins={n_bins}")
    except Exception as e:
        print(f"✗ Test failed with data={data}, n_bins={n_bins}")
        print(f"  Error: {type(e).__name__}: {e}")
        raise

# Run the test
print("Running hypothesis tests...")
print("=" * 60)
try:
    test_cut_coverage()
    print("\nAll hypothesis tests passed!")
except Exception as e:
    print(f"\nHypothesis test failed!")

# Now test the specific failing case
print("\n" + "=" * 60)
print("Testing the specific failing input from bug report:")
data = [1.1125369292536007e-308, -1.0]
n_bins = 2
print(f"data={data}, n_bins={n_bins}")
try:
    result = pd.cut(data, bins=n_bins)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")