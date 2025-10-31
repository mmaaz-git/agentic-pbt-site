#!/usr/bin/env python3

from Cython.Build.Dependencies import strip_string_literals

# Test what strip_string_literals does
test_cases = [
    "foo#bar",
    "[foo#bar]",
    "[libA, libB#version]",
    "foo#bar baz",
    "[#]",
    "# distutils: libraries = lib#version",
    "libraries = lib#version",
    "lib#version",
    "'lib#version'",
    '"lib#version"'
]

print("Testing strip_string_literals behavior:\n")

for test in test_cases:
    stripped, literals = strip_string_literals(test)
    print(f"Input:     {test!r}")
    print(f"Stripped:  {stripped!r}")
    print(f"Literals:  {literals}")
    print()