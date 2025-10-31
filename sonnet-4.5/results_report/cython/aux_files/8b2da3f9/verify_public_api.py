#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import _escape_special_type_characters, type_identifier_from_declaration, cap_length

failing_input = '000000000000000000000 0 0 0 0 '

# Test internal function
escaped = _escape_special_type_characters(failing_input)
print("Internal _escape_special_type_characters:")
print(f"  Input: '{failing_input}'")
print(f"  Output: '{escaped}'")
print(f"  Output length: {len(escaped)} (> 64)")
print()

# Test public API function
public_result = type_identifier_from_declaration(failing_input)
print("Public API type_identifier_from_declaration:")
print(f"  Input: '{failing_input}'")
print(f"  Output: '{public_result}'")
print(f"  Output length: {len(public_result)} (<= 64)")
print()

# Show how cap_length works
print("The public API works correctly because it applies cap_length after escaping:")
print(f"  cap_length(escaped): '{cap_length(escaped)}'")
print(f"  Length after capping: {len(cap_length(escaped))}")