#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import strip_string_literals

# Test with empty string in quotes
code = '""'
result, literals = strip_string_literals(code)
print(f"Input: {repr(code)}")
print(f"Result: {repr(result)}")
print(f"Literals: {literals}")

# The unquote logic would try:
if result and result[0] in '"\'':
    key = result[1:-1]
    print(f"Key to lookup: {repr(key)}")
    print(f"Key in literals? {key in literals}")