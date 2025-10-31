#!/usr/bin/env python3

from Cython.Build.Dependencies import DistutilsInfo, parse_list

# Test how distutils directives are parsed
source = """
# distutils: libraries = lib#version
# distutils: include_dirs = /path/with#hash
# cython: something = else
"""

print("Testing how DistutilsInfo parses directives with # characters:\n")
info = DistutilsInfo(source)
print(f"Parsed values: {info.values}")
print()

# Now test parse_list specifically
print("Testing parse_list with the value that would be extracted:\n")

# After extraction, the value would be "lib#version"
value = "lib#version"
print(f"Input value: {value!r}")
result = parse_list(value)
print(f"Result: {result}")
print()

value = "[lib#version, other#lib]"
print(f"Input value: {value!r}")
result = parse_list(value)
print(f"Result: {result}")
print()

# Test what happens before parse_list
from Cython.Build.Dependencies import strip_string_literals

# The actual directive line
directive_line = "# distutils: libraries = lib#version"
print(f"Original directive line: {directive_line!r}")
stripped, literals = strip_string_literals(directive_line)
print(f"After strip_string_literals: {stripped!r}")
print(f"Literals: {literals}")
print()

# What DistutilsInfo sees
source_with_directive = """
# distutils: libraries = lib#version
print("hello")
"""
print("Testing full source with directive:")
print(source_with_directive)
info = DistutilsInfo(source_with_directive)
print(f"Parsed libraries: {info.values.get('libraries', 'NOT FOUND')}")