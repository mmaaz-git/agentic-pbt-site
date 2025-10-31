#!/usr/bin/env python3
"""
Reproduction script for ARGSORT_DEFAULTS duplicate assignment bug.
"""

# Import the ARGSORT_DEFAULTS dictionary
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("=== ARGSORT_DEFAULTS Duplicate Assignment Bug ===")
print()

# Show the current value of ARGSORT_DEFAULTS
print("Current ARGSORT_DEFAULTS dictionary:")
print(f"  ARGSORT_DEFAULTS = {ARGSORT_DEFAULTS}")
print()

# Show the specific 'kind' value
print("Value of ARGSORT_DEFAULTS['kind']:")
print(f"  ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")
print()

# Demonstrate the issue
print("Issue in pandas/compat/numpy/function.py lines 138-140:")
print("  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
print("  Line 139: ARGSORT_DEFAULTS['order'] = None")
print("  Line 140: ARGSORT_DEFAULTS['kind'] = None  # <-- Overwrites line 138!")
print()

# Show what the expected behavior would be
print("Analysis:")
print("  - Line 138 sets 'kind' to 'quicksort'")
print("  - Line 140 immediately overwrites it with None")
print("  - Result: Line 138 is dead code with no effect")
print()

# Verify the final value
expected_from_line_138 = "quicksort"
actual_value = ARGSORT_DEFAULTS['kind']

print("Verification:")
print(f"  Expected from line 138: '{expected_from_line_138}'")
print(f"  Actual final value: {actual_value}")
print(f"  Match: {actual_value == expected_from_line_138}")
print()

if actual_value != expected_from_line_138:
    print("CONFIRMED: Line 138 is dead code - its assignment is immediately overwritten")
else:
    print("ERROR: Unexpected state - line 140 should have overwritten line 138")