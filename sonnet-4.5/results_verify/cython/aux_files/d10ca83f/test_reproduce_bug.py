#!/usr/bin/env python3
"""Test to reproduce the Cython.Plex.Regexps.RE.wrong_type bug"""

import sys

# First, let's verify the Python version
print(f"Python version: {sys.version}")
print(f"Python version info: {sys.version_info}")

# Test 1: Hypothesis test from the bug report
print("\n=== Test 1: Hypothesis test ===")
try:
    from hypothesis import given, strategies as st, settings
    import pytest
    from Cython.Plex import Seq, Str
    from Cython.Plex.Errors import PlexTypeError

    @given(st.text(alphabet='abc', min_size=1, max_size=5))
    @settings(max_examples=200)
    def test_seq_rejects_non_re_args(s):
        with pytest.raises(PlexTypeError):
            Seq(Str(s), "not an RE")

    # Run the test with a simple input
    print("Testing with s='a'...")
    try:
        seq = Seq(Str('a'), "not an RE")
        print("ERROR: Should have raised an exception!")
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")

except ImportError as e:
    print(f"ImportError: {e}")
    print("Skipping hypothesis test due to missing dependencies")

# Test 2: Direct reproduction from bug report
print("\n=== Test 2: Direct reproduction ===")
try:
    from Cython.Plex import Seq, Str

    print("Attempting to create Seq(Str('a'), 'not an RE')...")
    try:
        seq = Seq(Str('a'), "not an RE")
        print("ERROR: Should have raised an exception!")
    except AttributeError as e:
        print(f"AttributeError: {e}")
        print("Expected: PlexTypeError")
        print("✓ Bug reproduced - got AttributeError instead of PlexTypeError")
    except Exception as e:
        print(f"Got unexpected exception type: {type(e).__name__}")
        print(f"Exception message: {e}")

except Exception as e:
    print(f"Error during test: {e}")

# Test 3: Check if types.InstanceType exists
print("\n=== Test 3: Check types.InstanceType availability ===")
import types
print(f"Available attributes in types module: {[attr for attr in dir(types) if 'Instance' in attr]}")
try:
    types.InstanceType
    print("types.InstanceType exists (unexpected for Python 3)")
except AttributeError:
    print("✓ types.InstanceType does not exist in Python 3 (as expected)")
    print("This confirms the root cause of the bug")