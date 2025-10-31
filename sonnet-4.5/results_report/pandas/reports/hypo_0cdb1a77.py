#!/usr/bin/env python3
"""Hypothesis test for InfinityType comparison consistency."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import pandas.util.version as version_module

@given(st.sampled_from([
    version_module.Infinity,
    version_module.NegativeInfinity
]))
def test_comparison_reflexivity(x):
    """Test that comparison operators follow reflexivity property.

    Mathematical property: If x == x, then x <= x and x >= x must be True.
    """
    if x == x:
        assert x <= x, f"{x} should be <= itself when it equals itself"
        assert x >= x, f"{x} should be >= itself when it equals itself"

if __name__ == "__main__":
    # Run the test
    test_comparison_reflexivity()