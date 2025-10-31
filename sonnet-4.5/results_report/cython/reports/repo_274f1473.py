#!/usr/bin/env python3
"""Minimal reproduction of Cython GDB version detection bug."""

import re

# The regex from Cython.Debugger.Tests.TestLibCython.test_gdb() line 42
regex = r"GNU gdb [^\d]*(\d+)\.(\d+)"

# Real-world GDB version output from Ubuntu 22.04
gdb_output = "GNU gdb (Ubuntu 12.1-0ubuntu1~22.04) 7.2"

# Try to match the version
match = re.match(regex, gdb_output)
if match:
    version = list(map(int, match.groups()))
    print(f"Input string: {gdb_output}")
    print(f"Regex pattern: {regex}")
    print(f"Detected version: {version}")
    print(f"Expected: [7, 2] (actual GDB version)")
    print(f"Actual: {version} (Ubuntu package version)")

    # The bug: this assertion fails
    assert version == [7, 2], f"Bug: Matched Ubuntu package version {version} instead of GDB version [7, 2]"
else:
    print("No match found")