#!/usr/bin/env python3
"""Test script to reproduce the is_valid_tag bug"""

import tempfile
import sys
import os

# Add the cython env to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.DebugWriter import CythonDebugWriter, is_valid_tag
from Cython.Compiler.StringEncoding import EncodedString

print("=" * 60)
print("Testing is_valid_tag function")
print("=" * 60)

# Test 1: Direct reproduction from bug report
tag_name = EncodedString("0variable")
print(f"\nTest 1: Testing tag name '{tag_name}'")
print(f"is_valid_tag('{tag_name}') = {is_valid_tag(tag_name)}")

# Test with actual XML
print("\nAttempting to use the tag with XML writer...")
with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.module_name = "test"
    writer.start('Module')

    try:
        writer.start(tag_name)
        print("SUCCESS: XML accepted the tag")
    except ValueError as e:
        print(f"FAILURE: XML rejected the tag with error: {e}")

# Test 2: Test with other digit-starting tags
print("\n" + "=" * 60)
print("Test 2: Testing various digit-starting tags")
print("=" * 60)

for digit in range(10):
    tag = EncodedString(str(digit) + "tag")
    is_valid = is_valid_tag(tag)
    print(f"is_valid_tag('{tag}') = {is_valid}")

    # Try to use it with XML
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        writer.module_name = "test"
        writer.start('Module')

        try:
            writer.start(tag)
            print(f"  -> XML accepted '{tag}'")
        except Exception as e:
            print(f"  -> XML rejected '{tag}': {e}")

# Test 3: Check the intended case that should be invalid
print("\n" + "=" * 60)
print("Test 3: Testing the documented invalid case (.0, .1, etc)")
print("=" * 60)

for i in range(3):
    tag = EncodedString(f".{i}")
    is_valid = is_valid_tag(tag)
    print(f"is_valid_tag('{tag}') = {is_valid}")

# Test 4: Test with plain strings vs EncodedString
print("\n" + "=" * 60)
print("Test 4: Testing plain strings vs EncodedString")
print("=" * 60)

test_names = ["0tag", "tag0", "_tag", "tag", "9name", ".0"]
for name in test_names:
    plain_valid = is_valid_tag(name)
    encoded_valid = is_valid_tag(EncodedString(name))
    print(f"is_valid_tag('{name}'):")
    print(f"  plain string: {plain_valid}")
    print(f"  EncodedString: {encoded_valid}")