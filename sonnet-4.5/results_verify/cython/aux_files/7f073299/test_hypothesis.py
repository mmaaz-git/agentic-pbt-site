#!/usr/bin/env python3
"""Property-based test from the bug report"""

import sys
import tempfile

# Add the cython env to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

@given(st.integers(min_value=0, max_value=9))
def test_digit_starting_tags_should_be_invalid(digit):
    tag_name = EncodedString(str(digit) + "tag")

    # The test expects is_valid_tag to return True (which is the bug)
    assert is_valid_tag(tag_name) == True

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = "test"
        writer.start('Module')

        try:
            writer.start(tag_name)
            assert False, f"XML accepted tag '{tag_name}'"
        except ValueError:
            pass  # Expected - XML should reject digit-starting tags

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_digit_starting_tags_should_be_invalid()
        print("Test passed - demonstrating the inconsistency between is_valid_tag and XML validation")
    except Exception as e:
        print(f"Test failed: {e}")