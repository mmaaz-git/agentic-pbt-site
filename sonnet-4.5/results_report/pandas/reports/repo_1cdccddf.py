#!/usr/bin/env python3
"""Demonstrate the duplicate assignment bug in pandas ARGSORT_DEFAULTS"""

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

# Show the current value
print("Current ARGSORT_DEFAULTS dictionary:")
print(ARGSORT_DEFAULTS)
print()

# Show that 'kind' is None, not 'quicksort'
print(f"Value of ARGSORT_DEFAULTS['kind']: {ARGSORT_DEFAULTS['kind']}")
print(f"Type of ARGSORT_DEFAULTS['kind']: {type(ARGSORT_DEFAULTS['kind'])}")
print()

# According to numpy documentation, the default for 'kind' should be 'quicksort'
print("Expected value based on NumPy documentation: 'quicksort'")
print(f"Actual value: {ARGSORT_DEFAULTS['kind']}")
print()

# Check if the value equals what we expect
if ARGSORT_DEFAULTS['kind'] == 'quicksort':
    print("✓ The 'kind' key has the expected value 'quicksort'")
else:
    print("✗ BUG: The 'kind' key is None instead of 'quicksort'")
    print("  This is due to duplicate assignment in the source code:")
    print("  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
    print("  Line 140: ARGSORT_DEFAULTS['kind'] = None  # Overwrites the previous value")