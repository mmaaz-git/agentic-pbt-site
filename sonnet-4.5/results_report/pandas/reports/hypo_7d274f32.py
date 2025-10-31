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

if __name__ == "__main__":
    test_arrow_dtype_mapping_no_duplicate_keys()
    print("Test passed!")