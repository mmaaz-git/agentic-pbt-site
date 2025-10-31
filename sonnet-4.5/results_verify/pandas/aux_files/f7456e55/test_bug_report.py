#!/usr/bin/env python3

# First, let's test the Hypothesis test from the bug report
from hypothesis import given, strategies as st

def test_arrow_dtype_mapping_no_duplicate_keys():
    try:
        from pandas.io._util import _arrow_dtype_mapping
        import pyarrow as pa

        mapping = _arrow_dtype_mapping()

        string_count = len([k for k in mapping.keys() if k == pa.string()])

        assert string_count == 1, f"pa.string() appears {string_count} times"
        print(f"Test assertion: pa.string() appears {string_count} times")
        return string_count == 1
    except ImportError as e:
        print(f"ImportError: {e}")
        return None

# Run the hypothesis test
print("Running Hypothesis test...")
result = test_arrow_dtype_mapping_no_duplicate_keys()
if result is not None:
    print(f"Test {'PASSED' if result else 'FAILED'}")
else:
    print("Test skipped due to missing dependencies")

print("\n" + "="*50 + "\n")

# Now run the reproduction code
print("Running reproduction code...")
try:
    import pyarrow as pa
    import pandas as pd
    from pandas.io._util import _arrow_dtype_mapping

    mapping = _arrow_dtype_mapping()

    print("Dictionary literal from source code has duplicate key:")
    print("  Line 41: pa.string(): pd.StringDtype(),")
    print("  Line 44: pa.string(): pd.StringDtype(),")

    # Count actual occurrences
    string_keys = [k for k in mapping.keys() if k == pa.string()]
    actual_count = len(string_keys)
    print(f"\nActual: {actual_count} entry for pa.string() in resulting dictionary")

    # Check if both lines map to same value
    print(f"Value for pa.string(): {mapping.get(pa.string())}")

    # Demonstrate that line 44 is dead code
    print("\nLine 44 is effectively dead code - when Python creates the dict,")
    print("the second occurrence overwrites the first (both map to same value).")

    # Show all keys in the mapping
    print(f"\nTotal keys in mapping: {len(mapping)}")

except ImportError as e:
    print(f"Install with: pip install pyarrow (Error: {e})")