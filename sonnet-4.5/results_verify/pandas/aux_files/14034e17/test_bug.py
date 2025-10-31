#!/usr/bin/env python3
"""Test the reported bug in import_optional_dependency"""

from pandas.compat._optional import import_optional_dependency

# Test 1: Basic reproduction as described in the bug report
print("Test 1: Basic reproduction with numpy and min_version='999.0.0'")
result = import_optional_dependency("numpy", min_version="999.0.0", errors="ignore")
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print()

# Test 2: Verify numpy is actually installed and we can get it normally
print("Test 2: Import numpy without version requirement")
result_no_version = import_optional_dependency("numpy", errors="ignore")
print(f"Result without version requirement: {result_no_version}")
print(f"Module name: {result_no_version.__name__ if result_no_version else 'None'}")
print()

# Test 3: Test with errors='warn' for comparison
print("Test 3: Same test with errors='warn'")
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result_warn = import_optional_dependency("numpy", min_version="999.0.0", errors="warn")
    print(f"Result with warn: {result_warn}")
    if w:
        print(f"Warning raised: {w[0].message}")
print()

# Test 4: Test with a module that doesn't exist
print("Test 4: Non-existent module with errors='ignore'")
result_nonexistent = import_optional_dependency("nonexistent_module_xyz", errors="ignore")
print(f"Result for non-existent module: {result_nonexistent}")
print()

# Test 5: Test the actual implementation logic
print("Test 5: Trace through the implementation logic")
print("Looking at the code:")
print("- Line 135-139: If module cannot be imported, return None for errors='ignore'")
print("- Line 148-166: If version check fails and errors='ignore', line 166 returns None")
print("- Line 168: Return module")
print()
print("For our case with numpy installed but version too old:")
print("- Module imports successfully (passes line 135)")
print("- Version check fails (enters if block at line 151)")
print("- errors='ignore' branch at line 165-166 returns None")
print("- Never reaches line 168 to return the module")