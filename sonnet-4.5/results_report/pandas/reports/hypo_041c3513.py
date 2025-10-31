#!/usr/bin/env python3
"""
Property-based test showing that is_subperiod and is_superperiod are not inverse operations
"""

from hypothesis import given, settings, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    "D", "B", "C", "h", "min", "s", "ms", "us", "ns",
    "M", "BM", "W", "Y", "Q",
])

@settings(max_examples=1000)
@given(source=freq_strings, target=freq_strings)
def test_superperiod_subperiod_inverse(source, target):
    if is_superperiod(source, target):
        assert is_subperiod(target, source), (
            f"If {source} is_superperiod of {target}, "
            f"then {target} should be is_subperiod of {source}"
        )

if __name__ == "__main__":
    # Run the test to find failing cases
    test_superperiod_subperiod_inverse()