#!/usr/bin/env python3
"""Testing the py_version_hex function bug reported."""

from hypothesis import given, strategies as st, example
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Naming import py_version_hex

# First, let's test the property-based test from the bug report
@given(
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=255),
    st.integers(min_value=0, max_value=15),
    st.integers(min_value=0, max_value=15)
)
@example(3, 256, 0, 0, 0)  # The specific failing case mentioned
def test_py_version_hex_round_trip(major, minor, micro, level, serial):
    """Property: Version components should be extractable from hex value"""
    result = py_version_hex(major, minor, micro, level, serial)

    extracted_major = (result >> 24) & 0xFF
    extracted_minor = (result >> 16) & 0xFF
    extracted_micro = (result >> 8) & 0xFF
    extracted_level = (result >> 4) & 0xF
    extracted_serial = result & 0xF

    assert extracted_major == major, f"Major mismatch: expected {major}, got {extracted_major}"
    assert extracted_minor == minor, f"Minor mismatch: expected {minor}, got {extracted_minor}"
    assert extracted_micro == micro, f"Micro mismatch: expected {micro}, got {extracted_micro}"
    assert extracted_level == level, f"Level mismatch: expected {level}, got {extracted_level}"
    assert extracted_serial == serial, f"Serial mismatch: expected {serial}, got {extracted_serial}"

print("Running property-based test...")
try:
    test_py_version_hex_round_trip()
    print("✓ All property-based tests passed")
except AssertionError as e:
    print(f"✗ Property-based test failed: {e}")

# Now test the specific example from the bug report
print("\nTesting the specific examples from the bug report:")
v1 = py_version_hex(3, 0, 0)
v2 = py_version_hex(3, 256, 0)
print(f"py_version_hex(3, 0, 0)   = {hex(v1)}")
print(f"py_version_hex(3, 256, 0) = {hex(v2)}")
print(f"Same result? {v1 == v2}")

original_minor = 256
result = py_version_hex(3, original_minor, 0)
extracted_minor = (result >> 16) & 0xFF
print(f"Input minor: {original_minor}, Extracted minor: {extracted_minor}")
print(f"Data lost? {original_minor != extracted_minor}")

# Test with other overflow cases
print("\nTesting more overflow cases:")
test_cases = [
    (3, 300, 0, 0, 0),
    (256, 0, 0, 0, 0),
    (0, 0, 256, 0, 0),
    (0, 0, 0, 16, 0),
    (0, 0, 0, 0, 16),
]

for major, minor, micro, level, serial in test_cases:
    result = py_version_hex(major, minor, micro, level, serial)
    extracted = (
        (result >> 24) & 0xFF,
        (result >> 16) & 0xFF,
        (result >> 8) & 0xFF,
        (result >> 4) & 0xF,
        result & 0xF
    )
    print(f"Input: ({major}, {minor}, {micro}, {level}, {serial})")
    print(f"  Result hex: {hex(result)}")
    print(f"  Extracted: {extracted}")
    print(f"  Match: {(major, minor, micro, level, serial) == extracted}")