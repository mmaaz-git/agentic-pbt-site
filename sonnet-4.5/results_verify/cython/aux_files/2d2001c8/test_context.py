#!/usr/bin/env python3
"""Test what these functions generate - they seem to be C code generators"""

from Cython.Utility import pylong_join, _pylong_join

print("These functions generate C code for joining Python long integer digits.")
print("They're used to efficiently combine multiple digit values into a single value.\n")

print("For count=0 (no digits to join):")
print(f"  pylong_join(0):  {repr(pylong_join(0))}")
print(f"  _pylong_join(0): {repr(_pylong_join(0))}")
print()

print("In C, an empty expression '' would be invalid,")
print("while '()' is a valid (albeit unusual) expression that evaluates to void/nothing.")
print()

print("Example of generated code for count=3:")
print(f"  pylong_join(3):  {pylong_join(3)}")
print("This represents: (((d[2] << n) | d[1]) << n) | d[0]")
print()
print(f"  _pylong_join(3): {_pylong_join(3)[:100]}...")
print("This represents: (d[2] << 2*n) | (d[1] << 1*n) | d[0]")
print()

print("Both implement the same functionality (combining digit values) but with different shift patterns.")