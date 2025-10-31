#!/usr/bin/env python3
"""Test the CompatValidator bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import pytest
from pandas.compat.numpy.function import CompatValidator

# First, test with the hypothesis test
@given(method=st.text(min_size=1).filter(lambda x: x not in ["args", "kwargs", "both"]))
def test_compatvalidator_rejects_invalid_methods(method):
    validator = CompatValidator({}, method=method)
    with pytest.raises(ValueError, match="invalid validation method"):
        validator((), {})

# Manual reproduction
def test_manual():
    print("Testing manual reproduction:")
    validator = CompatValidator({}, method="invalid_method")

    # First call with empty args and kwargs
    print("First call with empty args and kwargs:")
    try:
        result = validator((), {})
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  ValueError raised: {e}")

    # Second call with non-empty args
    print("\nSecond call with non-empty args:")
    try:
        result = validator((1,), {})
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  ValueError raised: {e}")

if __name__ == "__main__":
    # First run the manual test
    test_manual()

    # Now run the hypothesis test by manually testing with '0'
    print("\nTesting with method='0':")
    validator = CompatValidator({}, method='0')
    try:
        with pytest.raises(ValueError, match="invalid validation method"):
            validator((), {})
        print("Test would pass with method='0' (exception was raised)")
    except AssertionError:
        print("Test failed with method='0': No exception was raised when one was expected")