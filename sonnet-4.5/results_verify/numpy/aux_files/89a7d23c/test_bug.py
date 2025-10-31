#!/usr/bin/env python3
"""Test the reported bug in numpy.ctypeslib.load_library"""

import tempfile
from unittest.mock import patch
import numpy as np
import traceback

# First, let's check what EXT_SUFFIX normally returns
import sysconfig
print(f"Normal EXT_SUFFIX value: {sysconfig.get_config_var('EXT_SUFFIX')}")

# Test 1: Reproduce the basic bug
print("\n=== Test 1: Basic reproduction ===")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('sysconfig.get_config_var', return_value=None):
            np.ctypeslib.load_library('mylib', tmpdir)
    print("No error occurred")
except TypeError as e:
    print(f"TypeError occurred: {e}")
    print(f"Error type matches bug report: {'unsupported operand' in str(e) or 'can only concatenate' in str(e)}")
except OSError as e:
    print(f"OSError occurred (expected): {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
    traceback.print_exc()

# Test 2: Run the property-based test
print("\n=== Test 2: Property-based test ===")
from hypothesis import given, settings, strategies as st

@given(
    libname=st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=10)
)
@settings(max_examples=10)  # Reduced for faster testing
def test_load_library_handles_none_ext_suffix(libname):
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch('sysconfig.get_config_var', return_value=None):
            try:
                np.ctypeslib.load_library(libname, tmpdir)
            except OSError as e:
                if "no file with expected extension" in str(e):
                    pass
                else:
                    raise
            except TypeError as e:
                if "unsupported operand type" in str(e) or "can only concatenate" in str(e):
                    assert False, f"Bug: load_library crashes when EXT_SUFFIX is None: {e}"
                else:
                    raise

try:
    test_load_library_handles_none_ext_suffix()
    print("Property test passed - no crashes detected")
except AssertionError as e:
    print(f"Property test failed: {e}")
except Exception as e:
    print(f"Unexpected error in property test: {e}")
    traceback.print_exc()

# Test 3: Check the exact line that's problematic
print("\n=== Test 3: Direct string concatenation test ===")
libname = "mylib"
so_ext = None
base_ext = ".so"

print(f"libname: {libname}")
print(f"so_ext: {so_ext}")
print(f"base_ext: {base_ext}")

# This is what the code tries to do
print("\nTrying: not so_ext == base_ext")
print(f"Result: {not so_ext == base_ext}")

print("\nTrying: libname + so_ext")
try:
    result = libname + so_ext
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    print("This is the exact error that occurs in load_library!")