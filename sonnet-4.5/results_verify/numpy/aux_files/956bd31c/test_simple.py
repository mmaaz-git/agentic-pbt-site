import numpy as np
import numpy.strings as nps

def test_replace_count_parameter(strings, count):
    """Test the nps.replace function with specific strings and count"""
    arr = np.array(strings)
    old = 'a'
    new = 'b'
    replaced = nps.replace(arr, old, new, count=count)

    for i, (original, result) in enumerate(zip(strings, replaced)):
        if count == -1:
            expected = original.replace(old, new)
        else:
            expected = original.replace(old, new, count)
        print(f"  Original: {repr(original)}, Result: {repr(result)}, Expected: {repr(expected)}")
        if result != expected:
            return False, f"Mismatch at index {i}: {repr(result)} != {repr(expected)}"
    return True, "All tests passed"

# Test the reported failing case
print("Testing the reported failing case: strings=['\x00'], count=0")
success, msg = test_replace_count_parameter(['\x00'], 0)
print(f"Result: {success}, {msg}")
print()

# The issue is actually in the array storage itself
print("The real issue - array storage of null characters:")
print("="*50)

# Test storing and retrieving null characters
test_strings = [
    '\x00',           # Single null
    'a\x00',          # Trailing null
    '\x00b',          # Leading null
    'a\x00b',         # Middle null
    '\x00\x00',       # Multiple nulls
    'test\x00\x00',   # String with multiple trailing nulls
]

for s in test_strings:
    arr = np.array([s])
    stored = arr[0]
    match = (stored == s)
    print(f"Input: {repr(s):20} -> Stored: {repr(stored):20} | Match: {match}")