#!/usr/bin/env python3
"""Demonstrate the Django check_referrer_policy bug with trailing comma"""

import sys
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from django.core.checks.security.base import REFERRER_POLICY_VALUES

# Test case 1: Trailing comma
test_string1 = "no-referrer,"
values1 = {v.strip() for v in test_string1.split(",")}

print("Test 1: Trailing comma")
print(f"Input: {test_string1!r}")
print(f"Parsed: {values1}")
print(f"Contains empty string: {'' in values1}")
print(f"Is valid subset: {values1 <= REFERRER_POLICY_VALUES}")
print()

# Test case 2: Double comma
test_string2 = "no-referrer,,same-origin"
values2 = {v.strip() for v in test_string2.split(",")}

print("Test 2: Double comma")
print(f"Input: {test_string2!r}")
print(f"Parsed: {values2}")
print(f"Contains empty string: {'' in values2}")
print(f"Is valid subset: {values2 <= REFERRER_POLICY_VALUES}")
print()

# Test case 3: Comma with only whitespace
test_string3 = "no-referrer, ,same-origin"
values3 = {v.strip() for v in test_string3.split(",")}

print("Test 3: Comma with only whitespace")
print(f"Input: {test_string3!r}")
print(f"Parsed: {values3}")
print(f"Contains empty string: {'' in values3}")
print(f"Is valid subset: {values3 <= REFERRER_POLICY_VALUES}")
print()

# Show valid values for reference
print("Valid REFERRER_POLICY_VALUES:")
for value in sorted(REFERRER_POLICY_VALUES):
    print(f"  - {value!r}")