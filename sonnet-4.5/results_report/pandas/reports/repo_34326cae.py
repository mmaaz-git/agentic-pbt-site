#!/usr/bin/env python3

from Cython.Build.Dependencies import DistutilsInfo

source_with_comment = """
# distutils: libraries = foo # this is a comment explaining foo
"""

info = DistutilsInfo(source_with_comment)
print("Parsed libraries:", info.values.get('libraries'))

# Show what was actually parsed
if info.values.get('libraries'):
    print("Number of libraries:", len(info.values.get('libraries')))
    for i, lib in enumerate(info.values.get('libraries')):
        print(f"  Library {i}: '{lib}'")

# Test the assertion
expected = ['foo']
actual = info.values.get('libraries')
print(f"\nExpected: {expected}")
print(f"Actual: {actual}")

if actual == expected:
    print("✓ Test passed: Libraries parsed correctly")
else:
    print("✗ Test failed: Libraries include comment text")
    if actual and len(actual) > 1 and actual[1].startswith('#'):
        print(f"  Bug confirmed: Comment transformed into label '{actual[1]}'")