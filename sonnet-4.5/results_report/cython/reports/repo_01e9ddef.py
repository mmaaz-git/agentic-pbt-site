#!/usr/bin/env python3
"""Minimal reproducer for py_version_hex integer overflow bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Naming import py_version_hex

# Test 1: Demonstrate that different inputs produce identical outputs
print("=== Test 1: Different inputs produce identical outputs ===")
v1 = py_version_hex(3, 0, 0)
v2 = py_version_hex(3, 256, 0)
print(f"py_version_hex(3, 0, 0)   = {hex(v1)}")
print(f"py_version_hex(3, 256, 0) = {hex(v2)}")
print(f"Same result? {v1 == v2}")
print()

# Test 2: Show that round-trip conversion fails
print("=== Test 2: Round-trip conversion fails ===")
original_minor = 256
result = py_version_hex(3, original_minor, 0)
# Extract the minor version from the hex result
extracted_minor = (result >> 16) & 0xFF
print(f"Original minor version: {original_minor}")
print(f"Encoded hex value: {hex(result)}")
print(f"Extracted minor version: {extracted_minor}")
print(f"Round-trip successful? {original_minor == extracted_minor}")
print()

# Test 3: Show multiple overflow cases
print("=== Test 3: Multiple overflow cases ===")
test_cases = [
    (3, 255, 0),  # Valid (max value)
    (3, 256, 0),  # Overflow by 1
    (3, 257, 0),  # Overflow by 2
    (3, 512, 0),  # Overflow by 256
    (256, 0, 0),  # Major version overflow
    (0, 0, 256),  # Micro version overflow
]

for major, minor, micro in test_cases:
    result = py_version_hex(major, minor, micro)
    extracted_major = (result >> 24) & 0xFF
    extracted_minor = (result >> 16) & 0xFF
    extracted_micro = (result >> 8) & 0xFF

    print(f"Input: ({major}, {minor}, {micro})")
    print(f"  Hex result: {hex(result)}")
    print(f"  Extracted: ({extracted_major}, {extracted_minor}, {extracted_micro})")
    if (major, minor, micro) != (extracted_major, extracted_minor, extracted_micro):
        print(f"  ERROR: Data corruption detected!")
    print()

# Test 4: Show collision - different inputs map to same output
print("=== Test 4: Collision demonstration ===")
colliding_inputs = [
    (3, 0, 0),
    (3, 256, 0),
    (3, 512, 0),
    (3, 768, 0),
]

results = []
for major, minor, micro in colliding_inputs:
    result = py_version_hex(major, minor, micro)
    results.append(result)
    print(f"py_version_hex({major}, {minor}, {micro}) = {hex(result)}")

print(f"\nAll produce the same output? {len(set(results)) == 1}")
print()

# Test 5: Release level and serial overflow
print("=== Test 5: Release level and serial overflow ===")
# Valid release level is 0-15, serial is 0-15
test_cases_level = [
    (3, 0, 0, 15, 15),  # Valid max values
    (3, 0, 0, 16, 0),   # Release level overflow
    (3, 0, 0, 0, 16),   # Release serial overflow
    (3, 0, 0, 255, 255), # Both overflow significantly
]

for major, minor, micro, level, serial in test_cases_level:
    result = py_version_hex(major, minor, micro, level, serial)
    extracted_level = (result >> 4) & 0xF
    extracted_serial = result & 0xF

    print(f"Input level={level}, serial={serial}")
    print(f"  Hex result: {hex(result)}")
    print(f"  Extracted level={extracted_level}, serial={extracted_serial}")
    if level != extracted_level or serial != extracted_serial:
        print(f"  ERROR: Data corruption in release fields!")
    print()