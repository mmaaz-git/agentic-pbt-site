#!/usr/bin/env python3
"""Test the reported bug in django.template.Variable"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.template import Variable

# Test case 1: Reproduction
print("=" * 60)
print("Test 1: Basic reproduction")
print("=" * 60)

var = Variable("10.")

print(f"var.var: {var.var}")
print(f"var.literal: {var.literal}")
print(f"var.lookups: {var.lookups}")

# Check the condition
if var.literal is not None and var.lookups is not None:
    print("ERROR: Both literal and lookups are set!")
else:
    print("OK: Only one of literal/lookups is set")

# Try to resolve it
try:
    result = var.resolve({})
    print(f"resolve() returned: {result}")
except Exception as e:
    print(f"resolve() raised {type(e).__name__}: {e}")

print()

# Test case 2: Python's float() behavior with trailing period
print("=" * 60)
print("Test 2: Python's float() behavior")
print("=" * 60)
try:
    python_float = float("10.")
    print(f"float('10.') = {python_float}")
except ValueError as e:
    print(f"float('10.') raised ValueError: {e}")

print()

# Test case 3: Test various similar cases
print("=" * 60)
print("Test 3: Various similar cases")
print("=" * 60)

test_cases = ["10", "10.0", "10.", "2.", "0.", "1.5", "1e5", "1.e5"]

for test_str in test_cases:
    try:
        var = Variable(test_str)
        print(f"'{test_str}': literal={var.literal}, lookups={var.lookups}")
        try:
            resolved = var.resolve({})
            print(f"  resolve() = {resolved}")
        except Exception as e:
            print(f"  resolve() raised {type(e).__name__}: {e}")
    except Exception as e:
        print(f"'{test_str}': Variable() raised {type(e).__name__}: {e}")

print()

# Test case 4: What happens with "." in the middle but not at end?
print("=" * 60)
print("Test 4: Period in middle vs at end")
print("=" * 60)

test_cases = ["foo.bar", "foo.", ".bar", "10.5", "10."]

for test_str in test_cases:
    try:
        var = Variable(test_str)
        print(f"'{test_str}': literal={var.literal}, lookups={var.lookups}")
    except Exception as e:
        print(f"'{test_str}': Variable() raised {type(e).__name__}: {e}")