#!/usr/bin/env python3
"""Test to reproduce the parse_timedelta bug with empty strings"""

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import dask.utils
import pytest
from hypothesis import given, strategies as st

def test_empty_string():
    """Test parse_timedelta with empty string"""
    print("Testing parse_timedelta with empty string...")
    try:
        result = dask.utils.parse_timedelta('')
        print(f"Result for empty string: {result}")
    except IndexError as e:
        print(f"IndexError raised: {e}")
    except ValueError as e:
        print(f"ValueError raised: {e}")
    except Exception as e:
        print(f"Other exception raised: {type(e).__name__}: {e}")

def test_space_only_string():
    """Test parse_timedelta with space-only string"""
    print("\nTesting parse_timedelta with space-only string...")
    try:
        result = dask.utils.parse_timedelta(' ')
        print(f"Result for space-only string: {result}")
    except IndexError as e:
        print(f"IndexError raised: {e}")
    except ValueError as e:
        print(f"ValueError raised: {e}")
    except Exception as e:
        print(f"Other exception raised: {type(e).__name__}: {e}")

def test_whitespace_strings():
    """Test parse_timedelta with various whitespace strings"""
    whitespace_cases = ['\r', '\n', '\t', '   ', '\r\n']

    for ws in whitespace_cases:
        print(f"\nTesting parse_timedelta with {repr(ws)}...")
        try:
            result = dask.utils.parse_timedelta(ws)
            print(f"Result for {repr(ws)}: {result}")
        except IndexError as e:
            print(f"IndexError raised: {e}")
        except ValueError as e:
            print(f"ValueError raised: {e}")
        except Exception as e:
            print(f"Other exception raised: {type(e).__name__}: {e}")

def test_valid_inputs():
    """Test parse_timedelta with valid inputs for comparison"""
    valid_cases = ['1s', '5ms', '3.5 seconds', '300ms', '2h']

    print("\n--- Testing valid inputs for comparison ---")
    for case in valid_cases:
        try:
            result = dask.utils.parse_timedelta(case)
            print(f"parse_timedelta({repr(case)}) = {result}")
        except Exception as e:
            print(f"parse_timedelta({repr(case)}) raised {type(e).__name__}: {e}")

def test_hypothesis_property():
    """Test the hypothesis property from the bug report"""
    print("\n--- Running hypothesis test ---")

    @given(st.just(''))
    def test_parse_timedelta_empty_string(s):
        with pytest.raises(ValueError):
            dask.utils.parse_timedelta(s)

    try:
        test_parse_timedelta_empty_string()
        print("Hypothesis test passed (ValueError was raised)")
    except AssertionError:
        print("Hypothesis test failed (ValueError was NOT raised)")
    except IndexError as e:
        print(f"Hypothesis test caused IndexError: {e}")
    except Exception as e:
        print(f"Hypothesis test caused other exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_empty_string()
    test_space_only_string()
    test_whitespace_strings()
    test_valid_inputs()
    test_hypothesis_property()