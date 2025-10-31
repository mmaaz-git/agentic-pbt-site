import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages")

# Test the actual runtime behavior
import pandas as pd
from pandas.io._util import _arrow_dtype_mapping

try:
    import pyarrow as pa
    mapping = _arrow_dtype_mapping()

    print(f"Total entries in mapping: {len(mapping)}")

    # Check if pa.string() is in the mapping
    string_key = pa.string()
    if string_key in mapping:
        print(f"pa.string() maps to: {mapping[string_key]}")

    # Count unique values
    print("\nAll mappings:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")

except ImportError as e:
    print(f"PyArrow not installed: {e}")
    print("\nTesting with a simple duplicate key dictionary:")

    # Demonstrate Python's behavior with duplicate keys
    test_dict = {
        "key1": "value1",
        "key2": "value2",
        "key1": "value3",  # This will overwrite the first "key1"
    }
    print(f"Dictionary with duplicate 'key1': {test_dict}")
    print(f"Length: {len(test_dict)}")
    print(f"Value of 'key1': {test_dict.get('key1')}")