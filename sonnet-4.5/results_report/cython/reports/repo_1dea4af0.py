#!/usr/bin/env python3
"""
Minimal reproduction of the Cython distutils directive comment handling bug.
This demonstrates how inline comments in distutils directives are incorrectly
included as placeholder labels in the parsed configuration values.
"""

import sys
# Add the Cython environment to the path
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages")

from Cython.Build.Dependencies import DistutilsInfo

# Test case 1: Simple inline comment
print("=" * 60)
print("Test 1: Simple inline comment")
print("=" * 60)
source1 = """
# distutils: libraries = foo # this is a comment
"""
info1 = DistutilsInfo(source1)
result1 = info1.values.get('libraries', [])
print(f"Source: '# distutils: libraries = foo # this is a comment'")
print(f"Expected: ['foo']")
print(f"Actual:   {result1}")
print()

# Test case 2: Multiple libraries with inline comment
print("=" * 60)
print("Test 2: Multiple libraries with inline comment")
print("=" * 60)
source2 = """
# distutils: libraries = foo bar # another comment
"""
info2 = DistutilsInfo(source2)
result2 = info2.values.get('libraries', [])
print(f"Source: '# distutils: libraries = foo bar # another comment'")
print(f"Expected: ['foo', 'bar']")
print(f"Actual:   {result2}")
print()

# Test case 3: List format with inline comment
print("=" * 60)
print("Test 3: List format with inline comment")
print("=" * 60)
source3 = """
# distutils: libraries = [foo, bar] # comment after list
"""
info3 = DistutilsInfo(source3)
result3 = info3.values.get('libraries', [])
print(f"Source: '# distutils: libraries = [foo, bar] # comment after list'")
print(f"Expected: ['foo', 'bar']")
print(f"Actual:   {result3}")
print()

# Test case 4: Include directories with comment
print("=" * 60)
print("Test 4: Include directories with comment")
print("=" * 60)
source4 = """
# distutils: include_dirs = /opt/include # path to headers
"""
info4 = DistutilsInfo(source4)
result4 = info4.values.get('include_dirs', [])
print(f"Source: '# distutils: include_dirs = /opt/include # path to headers'")
print(f"Expected: ['/opt/include']")
print(f"Actual:   {result4}")
print()

# Test the parse_list function directly
print("=" * 60)
print("Direct parse_list test:")
print("=" * 60)
from Cython.Build.Dependencies import parse_list

test_input = "foo # comment"
parsed = parse_list(test_input)
print(f"parse_list('foo # comment') = {parsed}")
print(f"Expected: ['foo']")
print()

# Show the problem: the placeholder label
if result1 and len(result1) > 1 and result1[1].startswith('#__Pyx_L'):
    print("=" * 60)
    print("BUG CONFIRMED: Placeholder labels are being included!")
    print(f"The second element '{result1[1]}' is a placeholder label")
    print("that should not be in the configuration values.")
    print("=" * 60)