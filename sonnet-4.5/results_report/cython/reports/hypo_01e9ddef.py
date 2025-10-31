#!/usr/bin/env python3
"""Hypothesis test for py_version_hex round-trip property"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from Cython.Compiler.Naming import py_version_hex

@given(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=15),
    st.integers(min_value=0, max_value=15)
)
@example(3, 256, 0, 0, 0)  # Add explicit failing example
@settings(max_examples=100)
def test_py_version_hex_round_trip(major, minor, micro, level, serial):
    """Property: Version components should be extractable from hex value"""
    result = py_version_hex(major, minor, micro, level, serial)

    extracted_major = (result >> 24) & 0xFF
    extracted_minor = (result >> 16) & 0xFF
    extracted_micro = (result >> 8) & 0xFF
    extracted_level = (result >> 4) & 0xF
    extracted_serial = result & 0xF

    assert extracted_major == major, f"Major mismatch: {major} != {extracted_major}"
    assert extracted_minor == minor, f"Minor mismatch: {minor} != {extracted_minor}"
    assert extracted_micro == micro, f"Micro mismatch: {micro} != {extracted_micro}"
    assert extracted_level == level, f"Level mismatch: {level} != {extracted_level}"
    assert extracted_serial == serial, f"Serial mismatch: {serial} != {extracted_serial}"

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for py_version_hex...")
    print("Testing that encoded version components can be extracted correctly.")
    print()

    try:
        test_py_version_hex_round_trip()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis demonstrates that py_version_hex fails the round-trip property:")
        print("Values encoded into the hex format cannot be reliably extracted back.")
        import traceback
        traceback.print_exc()