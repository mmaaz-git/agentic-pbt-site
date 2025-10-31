#!/usr/bin/env python3

# First, let's test the simple reproduction case
from Cython.Debugger.DebugWriter import is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

test_input = ".0"

result_str = is_valid_tag(test_input)
result_encoded = is_valid_tag(EncodedString(test_input))

print(f'is_valid_tag("{test_input}") = {result_str}')
print(f'is_valid_tag(EncodedString("{test_input}")) = {result_encoded}')

try:
    assert result_str == result_encoded
    print("PASS: Results are consistent")
except AssertionError:
    print(f"FAIL: Inconsistent results - str returns {result_str}, EncodedString returns {result_encoded}")

# Also test other similar inputs
for test_input in [".0", ".123", ".999", ".1", ".00"]:
    result_str = is_valid_tag(test_input)
    result_encoded = is_valid_tag(EncodedString(test_input))
    if result_str != result_encoded:
        print(f'Input "{test_input}": str={result_str}, EncodedString={result_encoded} - INCONSISTENT')
    else:
        print(f'Input "{test_input}": Both return {result_str} - OK')