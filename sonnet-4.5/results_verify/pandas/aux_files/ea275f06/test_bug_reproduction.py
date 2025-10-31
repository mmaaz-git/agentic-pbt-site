"""Test file to reproduce the _check_for_default_values bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.util._validators import _check_for_default_values
import pytest


class RaisesTypeErrorOnEq:
    """Object that raises TypeError when compared with =="""
    def __eq__(self, other):
        raise TypeError("Cannot compare")


@given(st.integers())
def test_check_for_default_values_crashes_on_typeerror(value):
    """
    Property: When comparison raises non-ValueError exceptions,
    the function should fall back to 'is' comparison, not crash.
    """
    obj = RaisesTypeErrorOnEq()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}

    # The bug report expects this to raise TypeError, proving the bug exists
    with pytest.raises(TypeError):
        _check_for_default_values('test_func', arg_val_dict, compat_args)


class UncomparableObject:
    def __eq__(self, other):
        raise TypeError("Cannot compare UncomparableObject")


def simple_reproduction():
    """Direct reproduction without hypothesis."""
    obj = UncomparableObject()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}

    try:
        _check_for_default_values('test_func', arg_val_dict, compat_args)
        print("No exception raised - bug might be fixed")
    except TypeError as e:
        print(f"TypeError raised as expected by bug report: {e}")
        return True
    except Exception as e:
        print(f"Different exception raised: {type(e).__name__}: {e}")
        return False

    return False


if __name__ == "__main__":
    print("Testing simple reproduction...")
    bug_exists = simple_reproduction()

    if bug_exists:
        print("\nBug confirmed: TypeError propagates instead of being caught")
    else:
        print("\nBug not confirmed")

    print("\nTesting with hypothesis...")
    try:
        test_check_for_default_values_crashes_on_typeerror(42)
        print("Hypothesis test passed (bug confirmed)")
    except AssertionError:
        print("Hypothesis test failed - bug might not exist")
    except Exception as e:
        print(f"Hypothesis test error: {e}")