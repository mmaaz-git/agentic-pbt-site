#!/usr/bin/env python3
"""Test to reproduce the key_split bug with invalid UTF-8 bytes"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

from hypothesis import given, strategies as st
from dask.utils import key_split

# First, let's test the specific failing case
print("Testing specific failing case: b'\\x80'")
invalid_bytes = b'\x80'
try:
    result = key_split(invalid_bytes)
    print(f"Result: {result}")
except UnicodeDecodeError as e:
    print(f"Crashed with UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Crashed with other exception: {e}")

print("\n" + "="*50 + "\n")

# Now let's test the property-based test
print("Running property-based test...")
@given(st.binary())
def test_key_split_bytes_idempotence(s):
    try:
        first_split = key_split(s)
        second_split = key_split(first_split)
        assert first_split == second_split, f"Not idempotent: {s!r} -> {first_split!r} -> {second_split!r}"
    except UnicodeDecodeError as e:
        print(f"UnicodeDecodeError on input {s!r}: {e}")
        raise
    except Exception as e:
        print(f"Other exception on input {s!r}: {e}")
        raise

# Run the test with a few iterations
try:
    test_key_split_bytes_idempotence()
    print("Property test passed")
except Exception as e:
    print(f"Property test failed: {e}")

print("\n" + "="*50 + "\n")

# Let's also test some other invalid UTF-8 sequences
print("Testing various invalid UTF-8 sequences:")
invalid_sequences = [
    b'\x80',  # Invalid start byte
    b'\xc0\x80',  # Overlong encoding
    b'\xff',  # Invalid byte
    b'\xed\xa0\x80',  # Surrogate half
    b'\xc2',  # Incomplete sequence
]

for seq in invalid_sequences:
    try:
        result = key_split(seq)
        print(f"{seq!r:20} -> {result!r}")
    except UnicodeDecodeError as e:
        print(f"{seq!r:20} -> UnicodeDecodeError: {str(e)[:50]}...")
    except Exception as e:
        print(f"{seq!r:20} -> Other exception: {e}")

print("\n" + "="*50 + "\n")

# Let's also test valid bytes for comparison
print("Testing valid UTF-8 bytes (for comparison):")
valid_sequences = [
    b'hello',
    b'hello-world-1',
    b'x-1-2-3',
    b'data',
]

for seq in valid_sequences:
    try:
        result = key_split(seq)
        print(f"{seq!r:20} -> {result!r}")
    except Exception as e:
        print(f"{seq!r:20} -> Exception: {e}")