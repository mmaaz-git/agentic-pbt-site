#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
from pandas.io.formats.format import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False,
                          allow_infinity=False), min_size=1, max_size=20))
@settings(max_examples=100)
def test_format_percentiles_no_rounding_to_zero_or_hundred(percentiles):
    assume(len(percentiles) > 0)
    assume(all(0 <= p <= 1 for p in percentiles))

    result = format_percentiles(percentiles)

    for i, (p, formatted) in enumerate(zip(percentiles, result)):
        if p != 0.0:
            assert formatted != "0%", f"Non-zero percentile {p} was rounded to 0%"
        if p != 1.0:
            assert formatted != "100%", f"Non-one percentile {p} was rounded to 100%"

# Run the test
print("Running hypothesis test...")
try:
    test_format_percentiles_no_rounding_to_zero_or_hundred()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")