#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("Current ARGSORT_DEFAULTS:")
print(ARGSORT_DEFAULTS)

print(f"\nActual 'kind' value: {ARGSORT_DEFAULTS['kind']!r}")
print(f"Expected: 'quicksort'")

# Run the test
def test_argsort_defaults_no_duplicate_keys():
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected kind='quicksort', got kind={ARGSORT_DEFAULTS['kind']!r}"

try:
    test_argsort_defaults_no_duplicate_keys()
    print("\nTest PASSED")
except AssertionError as e:
    print(f"\nTest FAILED: {e}")