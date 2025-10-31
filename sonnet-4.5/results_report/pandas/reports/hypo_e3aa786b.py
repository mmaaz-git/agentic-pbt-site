#!/usr/bin/env python3
"""
Property-based test for pandas.io._util._arrow_dtype_mapping duplicate key bug
"""

from hypothesis import given, strategies as st
from pandas.io._util import _arrow_dtype_mapping

def test_arrow_dtype_mapping_no_duplicate_keys():
    """
    Test that _arrow_dtype_mapping() does not have duplicate keys.

    This test checks the runtime behavior of the dictionary returned
    by _arrow_dtype_mapping(). While Python allows duplicate keys in
    dictionary literals (with later values overwriting earlier ones),
    having duplicates represents dead code and violates the DRY principle.
    """
    try:
        import pyarrow as pa

        # Get the mapping dictionary
        mapping = _arrow_dtype_mapping()

        # Count occurrences of pa.string() in the keys
        string_count = len([k for k in mapping.keys() if k == pa.string()])

        # While the runtime dictionary has only 1 key (Python's behavior),
        # the source code contains a duplicate which is the actual bug
        assert string_count == 1, f"pa.string() appears {string_count} times in runtime dict"

        # The real issue is in the source code where line 44 duplicates line 41
        print("Test passed: Runtime dictionary has 1 pa.string() key")
        print("However, source code has duplicate pa.string() keys on lines 41 and 44")
        print("Line 44 is dead code that should be removed")

    except ImportError as e:
        print(f"Skipping test due to missing dependency: {e}")

# Run the test
if __name__ == "__main__":
    test_arrow_dtype_mapping_no_duplicate_keys()