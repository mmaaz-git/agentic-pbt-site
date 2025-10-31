import numpy as np
import pandas as pd
from hypothesis import given, assume, settings
import hypothesis.extra.pandas as pdst
from hypothesis import strategies as st

@given(pdst.series(elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                   index=pdst.range_indexes(min_size=10, max_size=100)),
       st.integers(min_value=2, max_value=10))
@settings(max_examples=100)  # Run a reasonable number of tests
def test_rolling_mean_bounds(series, window_size):
    assume(len(series) >= window_size)

    rolling_mean = series.rolling(window=window_size).mean()

    failures = []
    for i in range(window_size - 1, len(series)):
        if not np.isnan(rolling_mean.iloc[i]):
            window_data = series.iloc[i - window_size + 1:i + 1]
            mean_val = rolling_mean.iloc[i]
            min_val = window_data.min()
            max_val = window_data.max()

            if not (mean_val >= min_val and mean_val <= max_val):
                failures.append({
                    'index': i,
                    'window_data': window_data.values,
                    'mean': mean_val,
                    'min': min_val,
                    'max': max_val
                })

    if failures:
        print(f"\nFound {len(failures)} failures out of {len(series) - window_size + 1} windows")
        for f in failures[:3]:  # Show first 3 failures
            print(f"  Index {f['index']}: mean={f['mean']}, min={f['min']}, max={f['max']}")
            print(f"    Window: {f['window_data']}")
        raise AssertionError(f"Mean not within bounds for {len(failures)} windows")

# Run the test
print("Running property-based test...")
try:
    test_rolling_mean_bounds()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")

# Also test with the specific failing case from the report
print("\nTesting specific failing case from report...")
series = pd.Series([-2.798597e-225, -2.225074e-308])
rolling_mean = series.rolling(window=2).mean()
print(f"Window: {series.values}")
print(f"Expected mean: {series.mean()}")
print(f"Rolling mean at index 1: {rolling_mean.iloc[1]}")
print(f"Mean within bounds? {series.min() <= rolling_mean.iloc[1] <= series.max()}")