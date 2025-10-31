import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from pandas.core.methods.describe import format_percentiles, describe_numeric_1d


@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100),
    st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=1, max_size=5).map(lambda x: sorted(list(set(x))))
)
@settings(max_examples=50, deadline=None)
def test_describe_numeric_min_max_bounds(data, percentiles):
    series = pd.Series(data)
    assume(series.count() > 0)
    assume(len(percentiles) > 0)

    try:
        result = describe_numeric_1d(series, percentiles)
        formatted_pcts = format_percentiles(percentiles)
        for pct_label in formatted_pcts:
            pct_val = float(result[pct_label])
    except ValueError as e:
        if "duplicate labels" in str(e):
            print(f"Found duplicate labels issue with percentiles: {percentiles}")
            formatted = format_percentiles(percentiles)
            print(f"Formatted: {formatted}")
            raise

# Run the test
print("Running property-based test...")
test_describe_numeric_min_max_bounds()