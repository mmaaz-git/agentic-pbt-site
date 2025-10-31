#!/usr/bin/env python3
"""Detailed investigation of the bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import os
from Cython.Debugger.Cygdb import make_command_file

prefix_code = '\r'
print(f"Input prefix_code: {repr(prefix_code)}")

result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)
print(f"Temp file created: {result}")

try:
    # Read in text mode (default)
    print("\n=== Reading with text mode (default) ===")
    with open(result, 'r') as f:
        content_text = f.read()
    print(f"Content starts with: {repr(content_text[:20])}")

    # Read in binary mode to see what's actually in the file
    print("\n=== Reading with binary mode ===")
    with open(result, 'rb') as f:
        content_binary = f.read()
    print(f"Binary content starts with: {repr(content_binary[:20])}")

    # Check what was actually written to the file
    first_byte = content_binary[0] if content_binary else None
    if first_byte == ord('\r'):
        print("The file actually contains \\r (0x0D)")
    elif first_byte == ord('\n'):
        print("The file actually contains \\n (0x0A)")
    else:
        print(f"The file starts with byte: {hex(first_byte)} ({chr(first_byte) if first_byte < 128 else 'non-ASCII'})")

    # Now let's verify the reading behavior
    print("\n=== Verifying read behavior ===")

    # Read with different newline modes
    print("Reading with newline=None (default - universal newlines):")
    with open(result, 'r', newline=None) as f:
        content1 = f.read(5)
    print(f"  Content: {repr(content1)}")

    print("Reading with newline='' (no translation):")
    with open(result, 'r', newline='') as f:
        content2 = f.read(5)
    print(f"  Content: {repr(content2)}")

    # Check if content preservation would work with newline=''
    print("\n=== Testing if reading with newline='' preserves \\r ===")
    with open(result, 'r', newline='') as f:
        content_notrans = f.read()
    if content_notrans.startswith('\r'):
        print("YES: Reading with newline='' preserves the \\r")
    else:
        print(f"NO: Even with newline='', content starts with {repr(content_notrans[0])}")

finally:
    if os.path.exists(result):
        os.remove(result)