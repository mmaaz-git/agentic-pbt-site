#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import _escape_special_type_characters

failing_input = '000000000000000000:<<,,,,'
result = _escape_special_type_characters(failing_input)

print(f"Input: '{failing_input}'")
print(f"Output: '{result}'")
print(f"Output length: {len(result)}")

# The test in TestTypes.py:75 expects this to be <= 64
try:
    assert len(result) <= 64, f"Length {len(result)} exceeds 64"
    print("Test PASSED: Length is within 64 characters")
except AssertionError as e:
    print(f"Test FAILED: {e}")
    print("\nThis demonstrates that _escape_special_type_characters can produce")
    print("output longer than 64 characters, violating the test's assertion.")