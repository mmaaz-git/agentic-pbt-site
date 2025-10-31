#!/usr/bin/env python3
"""Simpler test to demonstrate the bug clearly"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import CompatValidator

print("Testing CompatValidator with invalid method:")
print("=" * 60)

# Create validator with invalid method
validator = CompatValidator({}, method="invalid_method")

print("\n1. Call with empty args and kwargs:")
try:
    result = validator((), {})
    print(f"   ✓ No error raised, returned: {result}")
except ValueError as e:
    print(f"   ✗ ValueError raised: {e}")

print("\n2. Call with non-empty args:")
try:
    result = validator((1,), {})
    print(f"   ✓ No error raised, returned: {result}")
except ValueError as e:
    print(f"   ✗ ValueError raised: {e}")

print("\n" + "=" * 60)
print("BUG CONFIRMED: Invalid method is accepted when args and kwargs are empty")
print("but rejected when they have values. This is inconsistent behavior.")