#!/usr/bin/env python3
"""Reproduce the bug exactly as described in the report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import os
from Cython.Debugger.Cygdb import make_command_file

prefix_code = '\r'
result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)

try:
    with open(result, 'r') as f:
        content = f.read()

    print(f"Expected content to start with: {repr(prefix_code)}")
    print(f"Actual content starts with: {repr(content[:10])}")
    print(f"\nBug claim: \\r was converted to \\n")

    # Check what actually happened
    if len(content) > 0:
        print(f"First character is: {repr(content[0])}")
        if content[0] == '\n':
            print("Confirmed: First character is \\n, not \\r")
        elif content[0] == '\r':
            print("Actually: First character is still \\r")

    # The bug report's assertions
    try:
        assert content[0] == '\n', "First character is \\n, not \\r"
        print("Bug report's first assertion passed: content[0] == '\\n'")
    except AssertionError as e:
        print(f"Bug report's first assertion failed: {e}")

    try:
        assert content[0] != '\r', "Carriage return was lost"
        print("Bug report's second assertion passed: content[0] != '\\r'")
    except AssertionError as e:
        print(f"Bug report's second assertion failed: {e}")

finally:
    if os.path.exists(result):
        os.remove(result)