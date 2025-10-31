# Bug Report: Cython.Compiler.Naming.py_version_hex Integer Overflow Causes Silent Data Corruption

**Target**: `Cython.Compiler.Naming.py_version_hex`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `py_version_hex` function silently corrupts data when version components exceed their designated bit field sizes, causing different inputs to produce identical outputs and breaking round-trip conversion of encoded version numbers.

## Property-Based Test

```python
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
```

<details>

<summary>
**Failing input**: `major=3, minor=256, micro=0, level=0, serial=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 42, in <module>
    test_py_version_hex_round_trip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 11, in test_py_version_hex_round_trip
    st.integers(min_value=0, max_value=255),
               ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 30, in test_py_version_hex_round_trip
    assert extracted_minor == minor, f"Minor mismatch: {minor} != {extracted_minor}"
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Minor mismatch: 256 != 0
Falsifying explicit example: test_py_version_hex_round_trip(
    major=3,
    minor=256,
    micro=0,
    level=0,
    serial=0,
)
Running property-based test for py_version_hex...
Testing that encoded version components can be extracted correctly.

Test failed: Minor mismatch: 256 != 0

This demonstrates that py_version_hex fails the round-trip property:
Values encoded into the hex format cannot be reliably extracted back.
```
</details>

## Reproducing the Bug

```python
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
```

<details>

<summary>
Integer overflow causes multiple distinct version tuples to produce identical hex values
</summary>
```
=== Test 1: Different inputs produce identical outputs ===
py_version_hex(3, 0, 0)   = 0x3000000
py_version_hex(3, 256, 0) = 0x3000000
Same result? True

=== Test 2: Round-trip conversion fails ===
Original minor version: 256
Encoded hex value: 0x3000000
Extracted minor version: 0
Round-trip successful? False

=== Test 3: Multiple overflow cases ===
Input: (3, 255, 0)
  Hex result: 0x3ff0000
  Extracted: (3, 255, 0)

Input: (3, 256, 0)
  Hex result: 0x3000000
  Extracted: (3, 0, 0)
  ERROR: Data corruption detected!

Input: (3, 257, 0)
  Hex result: 0x3010000
  Extracted: (3, 1, 0)
  ERROR: Data corruption detected!

Input: (3, 512, 0)
  Hex result: 0x3000000
  Extracted: (3, 0, 0)
  ERROR: Data corruption detected!

Input: (256, 0, 0)
  Hex result: 0x100000000
  Extracted: (0, 0, 0)
  ERROR: Data corruption detected!

Input: (0, 0, 256)
  Hex result: 0x10000
  Extracted: (0, 1, 0)
  ERROR: Data corruption detected!

=== Test 4: Collision demonstration ===
py_version_hex(3, 0, 0) = 0x3000000
py_version_hex(3, 256, 0) = 0x3000000
py_version_hex(3, 512, 0) = 0x3000000
py_version_hex(3, 768, 0) = 0x3000000

All produce the same output? True

=== Test 5: Release level and serial overflow ===
Input level=15, serial=15
  Hex result: 0x30000ff
  Extracted level=15, serial=15

Input level=16, serial=0
  Hex result: 0x3000100
  Extracted level=0, serial=0
  ERROR: Data corruption in release fields!

Input level=0, serial=16
  Hex result: 0x3000010
  Extracted level=1, serial=0
  ERROR: Data corruption in release fields!

Input level=255, serial=255
  Hex result: 0x3000fff
  Extracted level=15, serial=15
  ERROR: Data corruption in release fields!

```
</details>

## Why This Is A Bug

This bug violates the fundamental contract of the `py_version_hex` function, which implements Python's PY_VERSION_HEX standard for encoding version numbers as 32-bit integers. The standard explicitly defines bit allocations for each component:

1. **Bits 31-24**: Major version (8 bits, valid range: 0-255)
2. **Bits 23-16**: Minor version (8 bits, valid range: 0-255)
3. **Bits 15-8**: Micro version (8 bits, valid range: 0-255)
4. **Bits 7-4**: Release level (4 bits, valid range: 0-15)
5. **Bits 3-0**: Release serial (4 bits, valid range: 0-15)

The bug manifests in several critical ways:

- **Silent Data Loss**: Values outside the valid ranges are silently truncated without warning. For example, `minor=256` becomes `minor=0` after encoding.
- **Collision Risk**: Multiple distinct version tuples map to the same hex value (e.g., `(3,0,0)`, `(3,256,0)`, `(3,512,0)` all produce `0x3000000`).
- **Round-Trip Failure**: Encoded values cannot be reliably decoded back to their original components, breaking the bijective property expected of an encoding function.
- **Standard Violation**: The function accepts and processes values that violate the PY_VERSION_HEX specification without validation.

## Relevant Context

The `py_version_hex` function is located in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Compiler/Naming.py:197-198` and is implemented as a simple bit-shifting operation without any input validation:

```python
def py_version_hex(major, minor=0, micro=0, release_level=0, release_serial=0):
    return (major << 24) | (minor << 16) | (micro << 8) | (release_level << 4) | (release_serial)
```

This function appears to be used internally by Cython for version comparisons and compatibility checks. While Python's actual version numbers will never exceed the valid ranges, the function's generic nature and lack of documentation about constraints make it prone to misuse. The Python C API documentation for `PY_VERSION_HEX` clearly specifies these constraints, which this implementation fails to enforce.

Related documentation:
- Python C API `PY_VERSION_HEX`: https://docs.python.org/3/c-api/apiabiversion.html#c.PY_VERSION_HEX
- Python's `sys.hexversion`: https://docs.python.org/3/library/sys.html#sys.hexversion

## Proposed Fix

Add input validation to ensure all version components are within their valid ranges before encoding:

```diff
 def py_version_hex(major, minor=0, micro=0, release_level=0, release_serial=0):
+    if not (0 <= major <= 255):
+        raise ValueError(f"major version must be 0-255, got {major}")
+    if not (0 <= minor <= 255):
+        raise ValueError(f"minor version must be 0-255, got {minor}")
+    if not (0 <= micro <= 255):
+        raise ValueError(f"micro version must be 0-255, got {micro}")
+    if not (0 <= release_level <= 15):
+        raise ValueError(f"release_level must be 0-15, got {release_level}")
+    if not (0 <= release_serial <= 15):
+        raise ValueError(f"release_serial must be 0-15, got {release_serial}")
     return (major << 24) | (minor << 16) | (micro << 8) | (release_level << 4) | (release_serial)
```