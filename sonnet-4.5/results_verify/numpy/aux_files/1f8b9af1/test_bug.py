#!/usr/bin/env python3
"""Test the reported bug in pandas.io.formats.printing.pprint_thing"""

import pandas.io.formats.printing as printing
import traceback

print("Test 1: Basic reproduction case from bug report")
print("=" * 60)
try:
    result = printing.pprint_thing("hello world", escape_chars=['a'])
    print(f"Success: {result!r}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Testing with standard escape characters (should work)")
try:
    result = printing.pprint_thing("hello\tworld\n", escape_chars=['\t', '\n'])
    print(f"Success with \\t and \\n: {result!r}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test 3: Testing with dict (should work for any character)")
try:
    result = printing.pprint_thing("hello world", escape_chars={'a': 'A'})
    print(f"Success with dict {'a': 'A'}: {result!r}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test 4: Testing with list containing mix of valid and invalid")
try:
    result = printing.pprint_thing("hello\tworld", escape_chars=['\t', 'o'])
    print(f"Success: {result!r}")
except KeyError as e:
    print(f"KeyError raised with mixed list: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 5: Testing empty string with custom escape char")
try:
    result = printing.pprint_thing("", escape_chars=['x'])
    print(f"Success with empty string: {result!r}")
except KeyError as e:
    print(f"KeyError raised with empty string: {e}")