#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Test the exact bug as described
arg_string = "py:my_var"

name = arg_string
if name.startswith('py:'):
    name = name[:3]  # This is the buggy line

print(f"Input argument: py:my_var=42")
print(f"Expected variable name: 'my_var'")
print(f"Actual variable name: {name!r}")
print(f"Bug: All py: arguments are stored as vars['py:'] instead of vars['my_var']")
print()

# Test with the corrected version
name2 = arg_string
if name2.startswith('py:'):
    name2 = name2[3:]  # This is the fix

print("With the fix:")
print(f"Input argument: py:my_var=42")
print(f"Expected variable name: 'my_var'")
print(f"Actual variable name: {name2!r}")
print(f"Result: Variable is correctly stored as vars['my_var']")