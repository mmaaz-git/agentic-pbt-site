import numpy as np
import numpy.char as char

# Test case with German eszett that expands when uppercased
arr = np.array(['ß'], dtype=str)
result = char.upper(arr)

print(f"Input: {arr[0]!r} (dtype: {arr.dtype})")
print(f"Result: {result[0]!r} (dtype: {result.dtype})")
print(f"Expected (Python str.upper): {'ß'.upper()!r}")

# Show the mismatch
try:
    assert result[0] == 'SS', f"Expected 'SS', got {result[0]!r}"
except AssertionError as e:
    print(f"\nAssertion failed: {e}")

# Test additional Unicode characters that expand
print("\n--- Additional examples of truncation ---")
test_cases = [
    'ﬁ',  # Latin Small Ligature Fi -> FI
    'ﬂ',  # Latin Small Ligature Fl -> FL
    'ﬃ', # Latin Small Ligature Ffi -> FFI
    'ﬄ', # Latin Small Ligature Ffl -> FFL
    'ﬅ',  # Latin Small Ligature Long S T -> ST
    'ﬆ',  # Latin Small Ligature St -> ST
]

for char_input in test_cases:
    arr = np.array([char_input], dtype=str)
    result = char.upper(arr)
    expected = char_input.upper()
    print(f"Input: {char_input!r} -> numpy: {result[0]!r}, expected: {expected!r}")