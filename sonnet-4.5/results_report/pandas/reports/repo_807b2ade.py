#!/usr/bin/env python3
"""
Minimal reproduction case for Cython.Debugger.DebugWriter.is_valid_tag bug.
This demonstrates the type-dependent behavior where the function returns different
results for the same logical input depending on whether it's a str or EncodedString.
"""

from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

# Test with the failing input ".0"
test_input = ".0"

# Test with regular string
result_str = is_valid_tag(test_input)
print(f'is_valid_tag("{test_input}") = {result_str}')

# Test with EncodedString
result_encoded = is_valid_tag(EncodedString(test_input))
print(f'is_valid_tag(EncodedString("{test_input}")) = {result_encoded}')

# These should be equal, but they are not
print(f"\nInconsistency detected: {result_str} != {result_encoded}")

# Test with a few more similar patterns
for test in [".123", ".999", ".1", ".00"]:
    str_result = is_valid_tag(test)
    enc_result = is_valid_tag(EncodedString(test))
    print(f'\nInput: "{test}"')
    print(f"  Regular string: {str_result}")
    print(f"  EncodedString:  {enc_result}")
    print(f"  Match: {str_result == enc_result}")

# This assertion will fail, demonstrating the bug
assert result_str == result_encoded, \
    f"Type-dependent behavior: is_valid_tag({test_input!r}) returns {result_str}, " \
    f"but is_valid_tag(EncodedString({test_input!r})) returns {result_encoded}"