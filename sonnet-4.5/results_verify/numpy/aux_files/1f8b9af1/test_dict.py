#!/usr/bin/env python3
"""Test dict escape_chars behavior"""

import pandas.io.formats.printing as printing
import traceback

# Test with simple dict replacement
print("Test with dict escape_chars:")
print("-" * 40)
try:
    result = printing.pprint_thing("hello world", escape_chars={'o': 'X', 'l': 'Y'})
    print(f"Input: 'hello world'")
    print(f"escape_chars: {{'o': 'X', 'l': 'Y'}}")
    print(f"Result: {result!r}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "-" * 40)
print("Test with dict containing standard escapes:")
try:
    result = printing.pprint_thing("hello\tworld\n", escape_chars={'\t': '\\t', '\n': '\\n'})
    print(f"Input: 'hello\\tworld\\n'")
    print(f"escape_chars: {{'\\t': '\\\\t', '\\n': '\\\\n'}}")
    print(f"Result: {result!r}")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()