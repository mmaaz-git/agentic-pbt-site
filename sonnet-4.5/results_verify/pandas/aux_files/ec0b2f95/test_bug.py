#!/usr/bin/env python3
"""Test the duplicate dictionary key bug in pandas.io._util._arrow_dtype_mapping"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

# First, let's run the hypothesis test from the bug report
from hypothesis import given, strategies as st
import pandas.io._util as util

def test_arrow_dtype_mapping_no_duplicate_keys():
    pa = util.import_optional_dependency("pyarrow")
    mapping = util._arrow_dtype_mapping()

    expected_unique_keys = 14
    actual_keys = len(mapping)

    assert actual_keys == expected_unique_keys, (
        f"Expected {expected_unique_keys} keys but got {actual_keys}. "
        "Duplicate keys detected in dictionary literal."
    )

print("Running Hypothesis test...")
try:
    test_arrow_dtype_mapping_no_duplicate_keys()
    print("Hypothesis test PASSED (no duplicate keys found)")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Hypothesis test ERROR: {e}")

print("\n" + "="*60 + "\n")

# Now run the reproduction code from the bug report
print("Running reproduction code...")
try:
    pa = util.import_optional_dependency("pyarrow")
    mapping = util._arrow_dtype_mapping()

    print(f"Number of keys: {len(mapping)}")
    print(f"Expected: 14 unique keys")
    print(f"Actual: 13 keys (pa.string() appears twice, second overwrites first)")

    # Let's also check what types are in the mapping
    print("\nKeys in the mapping:")
    for i, key in enumerate(mapping.keys(), 1):
        print(f"  {i}. {key}")

    # Verify that both pa.string() values map to the same thing
    print(f"\nValue for pa.string(): {mapping[pa.string()]}")

    # Check if large_string is present
    print(f"Value for pa.large_string(): {mapping[pa.large_string()]}")

except Exception as e:
    print(f"Reproduction code ERROR: {e}")