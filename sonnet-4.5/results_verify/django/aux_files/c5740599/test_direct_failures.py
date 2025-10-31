#!/usr/bin/env python3
"""Direct test of the reported failures"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.signing import b62_encode, b62_decode

def test_round_trip(s):
    """Test round-trip for a single string"""
    decoded = b62_decode(s)
    re_encoded = b62_encode(decoded)
    success = (re_encoded == s)
    print(f"Input: '{s}'")
    print(f"  Decoded: {decoded}")
    print(f"  Re-encoded: '{re_encoded}'")
    print(f"  Round-trip success: {success}")
    return success

# Test the specific cases from the bug report
test_cases = ['-', '-0', '0', '-5', '-A', 'ABC', '-ABC', '123']

print("Testing round-trip property for specific cases:")
print("=" * 60)

failures = []
for test_case in test_cases:
    success = test_round_trip(test_case)
    if not success:
        failures.append(test_case)
    print()

if failures:
    print(f"FAILURES FOUND: {failures}")
else:
    print("All tests passed")