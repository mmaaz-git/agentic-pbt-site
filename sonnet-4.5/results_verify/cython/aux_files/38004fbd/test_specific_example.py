#!/usr/bin/env python3
"""Test the specific example from the bug report"""
import sys
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/site-packages')

from Cython.Debugger.DebugWriter import is_valid_tag, CythonDebugWriter

# Test the reported failing case
tag = "0"
print(f"is_valid_tag('{tag}') = {is_valid_tag(tag)}")

with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    try:
        writer.start(tag)
        print(f"XML accepted '{tag}' as a valid tag")
    except ValueError as e:
        print(f"ValueError: {e}")

# Test other reported failing cases
test_cases = ['0', '123', '-invalid', 'with space', 'with:colon', '.', '']
for test_tag in test_cases:
    is_valid = is_valid_tag(test_tag)
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = CythonDebugWriter(tmpdir)
        try:
            writer.start(test_tag)
            xml_accepts = True
        except ValueError:
            xml_accepts = False

    if is_valid != xml_accepts:
        print(f"MISMATCH: is_valid_tag({test_tag!r}) = {is_valid}, XML accepts = {xml_accepts}")