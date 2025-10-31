#!/usr/bin/env python3
"""Test the proposed fix for the bug"""

import tempfile
from unittest.mock import patch
import numpy as np

# Create a mock version with the fix
def load_library_fixed(libname, loader_path):
    """Simulated fixed version of load_library"""
    import os
    import sys
    import sysconfig
    import ctypes

    # Convert path-like objects into strings
    libname = os.fsdecode(libname)
    loader_path = os.fsdecode(loader_path)

    ext = os.path.splitext(libname)[1]
    if not ext:
        # Try to load library with platform-specific name
        base_ext = ".so"
        if sys.platform.startswith("darwin"):
            base_ext = ".dylib"
        elif sys.platform.startswith("win"):
            base_ext = ".dll"
        libname_ext = [libname + base_ext]
        so_ext = sysconfig.get_config_var("EXT_SUFFIX")

        # THE FIX: Check if so_ext is not None before using it
        if so_ext is not None and so_ext != base_ext:
            libname_ext.insert(0, libname + so_ext)
    else:
        libname_ext = [libname]

    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path

    for ln in libname_ext:
        libpath = os.path.join(libdir, ln)
        if os.path.exists(libpath):
            try:
                return ctypes.cdll[libpath]
            except OSError:
                raise

    raise OSError("no file with expected extension")

# Test the fixed version
print("=== Testing fixed version ===")
with tempfile.TemporaryDirectory() as tmpdir:
    with patch('sysconfig.get_config_var', return_value=None):
        try:
            load_library_fixed('mylib', tmpdir)
            print("Fixed version raised OSError as expected")
        except OSError as e:
            if "no file with expected extension" in str(e):
                print(f"✓ Fixed version properly raises OSError: {e}")
            else:
                print(f"✗ Unexpected OSError: {e}")
        except TypeError as e:
            print(f"✗ Fixed version still has TypeError: {e}")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")

# Test that normal operation still works with the fix
print("\n=== Testing that fix doesn't break normal operation ===")
libname = "mylib"
so_ext = ".cpython-313-x86_64-linux-gnu.so"  # Normal value
base_ext = ".so"

# Check the original condition
original_condition = not so_ext == base_ext
print(f"Original condition (not so_ext == base_ext): {original_condition}")

# Check the fixed condition
fixed_condition = so_ext is not None and so_ext != base_ext
print(f"Fixed condition (so_ext is not None and so_ext != base_ext): {fixed_condition}")

print(f"Both conditions give same result for normal case: {original_condition == fixed_condition}")

# Test edge cases
print("\n=== Testing edge cases ===")
test_cases = [
    (None, ".so", "None vs .so"),
    ("", ".so", "empty string vs .so"),
    (".so", ".so", ".so vs .so"),
    (".dylib", ".so", ".dylib vs .so"),
]

for so_ext, base_ext, desc in test_cases:
    try:
        original = not so_ext == base_ext
        fixed = so_ext is not None and so_ext != base_ext
        print(f"{desc}:")
        print(f"  Original: {original}, Fixed: {fixed}")
        if so_ext is None:
            # Try concatenation
            try:
                _ = "lib" + so_ext
                print(f"  Concatenation: works")
            except TypeError:
                print(f"  Concatenation: TypeError (would crash)")
    except Exception as e:
        print(f"  Error: {e}")