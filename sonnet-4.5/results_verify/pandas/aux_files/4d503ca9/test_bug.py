#!/usr/bin/env python3
"""Test script to reproduce the _split_line bug."""

from pandas.io.sas.sas_xport import _split_line

# Test case without '_' in parts
parts = [('name', 5), ('value', 3)]
s = 'Alice123'

try:
    result = _split_line(s, parts)
    print(f"Result: {result}")
except KeyError as e:
    print(f"KeyError occurred: {e}")
    print("Bug confirmed: _split_line raises KeyError when '_' is not in parts")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")