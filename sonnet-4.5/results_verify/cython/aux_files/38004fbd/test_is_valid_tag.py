#!/usr/bin/env python3
"""Test to reproduce the is_valid_tag bug"""
import sys
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import is_valid_tag, CythonDebugWriter

@given(st.text())
@settings(max_examples=100)  # Reduced for quick test
def test_is_valid_tag_matches_xml_validity(tag_name):
    is_valid_result = is_valid_tag(tag_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        try:
            writer.start(tag_name)
            xml_accepts = True
        except ValueError:
            xml_accepts = False

    if is_valid_result and not xml_accepts:
        raise AssertionError(
            f"is_valid_tag({tag_name!r}) returned True, "
            f"but XML rejected it as invalid tag"
        )

# Run the test
if __name__ == "__main__":
    test_is_valid_tag_matches_xml_validity()
    print("Test completed")