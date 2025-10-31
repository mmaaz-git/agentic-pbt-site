#!/usr/bin/env python3
"""
Minimal reproduction case for pandas.io._util._arrow_dtype_mapping duplicate key bug
"""

try:
    import pyarrow as pa
    import pandas as pd
    from pandas.io._util import _arrow_dtype_mapping

    # Get the mapping dictionary
    mapping = _arrow_dtype_mapping()

    print("=== Duplicate Dictionary Key Bug Demonstration ===\n")
    print("Source code in pandas/io/_util.py contains:")
    print("  Line 41: pa.string(): pd.StringDtype(),")
    print("  Line 44: pa.string(): pd.StringDtype(),")
    print("\nThis creates a duplicate key in the dictionary literal.\n")

    # Count occurrences of pa.string() in the keys
    string_key_count = len([k for k in mapping.keys() if k == pa.string()])
    print(f"Number of pa.string() keys in resulting dictionary: {string_key_count}")

    # Show that line 44 is dead code
    print("\nPython behavior with duplicate dictionary keys:")
    print("  - Later values overwrite earlier ones")
    print("  - Line 44 overwrites line 41 with the same value")
    print("  - Result: Line 44 is effectively dead code")

    # Verify both string types are present
    has_string = pa.string() in mapping
    has_large_string = pa.large_string() in mapping

    print(f"\npa.string() in mapping: {has_string}")
    print(f"pa.large_string() in mapping: {has_large_string}")

    # Show the actual mapping
    print(f"\nmapping[pa.string()] = {mapping[pa.string()]}")
    print(f"mapping[pa.large_string()] = {mapping[pa.large_string()]}")

    print("\n=== Conclusion ===")
    print("The duplicate pa.string() key on line 44 should be removed.")
    print("This is dead code that violates the DRY principle.")

except ImportError as e:
    print(f"Error: {e}")
    print("Please install pyarrow: pip install pyarrow")