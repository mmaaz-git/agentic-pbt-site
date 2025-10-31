#!/usr/bin/env python3
"""Property-based test for the File.open() bug"""

from django.core.files.base import File
from io import BytesIO
from hypothesis import given, strategies as st, settings
import tempfile
import os

def test_file_reopen_without_mode(content):
    """Test that File.open() raises AttributeError when mode is missing"""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(content)
        temp_path = tf.name

    try:
        f = File(BytesIO(content), name=temp_path)
        f.close()

        try:
            f.open()
            # If we get here, the bug is fixed
            print(f"✗ FAIL: Expected AttributeError, but file opened successfully for content size {len(content)}")
            return False
        except AttributeError as e:
            # Bug occurred as expected
            assert 'mode' in str(e), f"AttributeError message doesn't contain 'mode': {e}"
            # This is the bug we're testing for
            return True
        except Exception as e:
            print(f"✗ FAIL: Unexpected error type {type(e).__name__}: {e}")
            return False
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    print("Running property-based test...")
    print("-" * 40)

    test_count = 0
    bug_count = 0

    # Run the test with several examples
    test_cases = [
        b'',  # Empty
        b'a',  # Single byte
        b'test content',  # Regular content
        b'\x00\x01\x02\x03',  # Binary content
        b'a' * 1000,  # Large content
    ]

    for i, content in enumerate(test_cases):
        test_count += 1
        print(f"Test {i+1}: content size = {len(content)} bytes")
        if test_file_reopen_without_mode(content):
            bug_count += 1
            print(f"  ✓ Bug reproduced (AttributeError with 'mode' raised)")
        else:
            print(f"  ✗ Bug not reproduced")

    print("-" * 40)
    print(f"Results: {bug_count}/{test_count} tests reproduced the bug")

    if bug_count == test_count:
        print("✓ Bug consistently reproduced across all test cases")