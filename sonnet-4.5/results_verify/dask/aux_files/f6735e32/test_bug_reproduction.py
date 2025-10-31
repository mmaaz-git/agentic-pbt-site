#!/usr/bin/env python3
"""Test to reproduce the bug in _normalize_and_strip_protocol function"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example

# First, let's reproduce the specific failing cases
def test_specific_failing_cases():
    from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

    print("Testing specific failing cases:")

    # Test case 1: "/"
    result = _normalize_and_strip_protocol("/")
    print(f"Input: '/' -> Result: {result}")
    assert result == [""], f"Expected [''], got {result}"

    # Test case 2: "///"
    result = _normalize_and_strip_protocol("///")
    print(f"Input: '///' -> Result: {result}")
    assert result == [""], f"Expected [''], got {result}"

    # Test case 3: "file:///"
    result = _normalize_and_strip_protocol("file:///")
    print(f"Input: 'file:///' -> Result: {result}")
    assert result == [""], f"Expected [''], got {result}"

    # Test case 4: "s3:///"
    result = _normalize_and_strip_protocol("s3:///")
    print(f"Input: 's3:///' -> Result: {result}")
    assert result == [""], f"Expected [''], got {result}"

    print("All specific test cases reproduced successfully!\n")


# Property-based test as provided in the bug report
@given(st.lists(st.text(min_size=0), min_size=1))
@example(["/"])
@example(["///"])
@example(["file:///"])
@example(["s3:///"])
def test_no_empty_strings_in_normalized_paths(paths):
    from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol
    result = _normalize_and_strip_protocol(paths)
    for r in result:
        assert r != "", f"Empty string found in result for input {paths}"


if __name__ == "__main__":
    # Test the specific cases first
    test_specific_failing_cases()

    # Run the property-based test
    print("Running property-based test...")
    try:
        test_no_empty_strings_in_normalized_paths()
        print("Property-based test passed (no empty strings found)!")
    except AssertionError as e:
        print(f"Property-based test failed: {e}")