#!/usr/bin/env python3
"""
Minimal reproduction of the CyLocals.invoke bug.
This demonstrates that calling max() on an empty dictionary raises ValueError.
"""

# Simulate the code path in CyLocals.invoke method (line 1262)
local_cython_vars = {}  # Empty dictionary, as would occur with a function having no locals

try:
    # This is the exact line from libcython.py:1262
    max_name_length = len(max(local_cython_vars, key=len))
    print(f"max_name_length: {max_name_length}")
except ValueError as e:
    print(f"ValueError: {e}")
    print("\nThis error occurs in CyLocals.invoke when a Cython function has no local variables.")