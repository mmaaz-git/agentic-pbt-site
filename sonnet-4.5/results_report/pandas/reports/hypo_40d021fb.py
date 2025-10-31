#!/usr/bin/env python3
"""Hypothesis test for pandas.tseries.frequencies reflexivity bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.tseries.frequencies import is_subperiod, is_superperiod

freq_strings = st.sampled_from([
    'D', 'h', 'min', 's', 'ms', 'us', 'ns', 'B', 'C',
    'W', 'W-MON', 'M', 'BM', 'Q', 'Q-JAN',
    'Y', 'Y-JAN',
])

@given(freq_strings)
def test_reflexivity_consistency(freq):
    """Test that is_superperiod and is_subperiod have consistent reflexivity."""
    is_super = is_superperiod(freq, freq)
    is_sub = is_subperiod(freq, freq)
    assert is_super == is_sub, f"Inconsistent reflexivity for {freq}: is_superperiod={is_super}, is_subperiod={is_sub}"

if __name__ == "__main__":
    test_reflexivity_consistency()