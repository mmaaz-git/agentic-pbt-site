import numpy as np

# Test the documented behavior
s1 = np.str_("abc\x00")
print(f"np.str_('abc\\x00') = {repr(s1)}")
print(f"Length: {len(s1)}")
print()

# Test various null positions
test_cases = [
    "a\x00",        # Trailing null
    "\x00a",        # Leading null
    "a\x00b",       # Middle null
    "a\x00\x00",    # Multiple trailing nulls
    "a\x00b\x00",   # Middle and trailing null
]

for test in test_cases:
    np_str = np.str_(test)
    py_str = test
    print(f"Input: {repr(test):15} -> np.str_: {repr(np_str):10} (len={len(np_str)}), Python: {repr(py_str):10} (len={len(py_str)})")