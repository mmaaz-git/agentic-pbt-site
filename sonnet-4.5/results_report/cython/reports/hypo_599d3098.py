#!/usr/bin/env python3
"""Property-based test for Cython TypeSlots error handling."""

from hypothesis import given, strategies as st
from Cython.Compiler.TypeSlots import get_slot_by_name
import pytest


@given(st.text(min_size=1, max_size=30))
def test_get_slot_error_type(slot_name):
    try:
        get_slot_by_name(slot_name, {})
    except AssertionError:
        pytest.fail("Bug: AssertionError instead of proper exception")
    except (ValueError, KeyError, LookupError):
        pass


if __name__ == "__main__":
    # Run the test to find a failing case
    test_get_slot_error_type()