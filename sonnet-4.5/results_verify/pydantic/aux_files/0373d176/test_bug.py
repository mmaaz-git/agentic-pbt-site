#!/usr/bin/env python3
"""Test script to reproduce the reported bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.alias_generators import to_snake

# Test the specific example from the bug report
print("Testing specific example from bug report:")
field = 'A0'
once = to_snake(field)
twice = to_snake(once)

print(f"to_snake('{field}') = '{once}'")
print(f"to_snake('{once}') = '{twice}'")
print(f"Expected: '{once}' == '{twice}'")
print(f"Actual: '{once}' {'==' if once == twice else '!='} '{twice}'")
print()

# Test with more examples to understand the behavior
test_cases = ['A0', 'a0', 'a_0', 'AB0', 'Ab0', 'HTTPResponse', 'myVar123', 'snake_case', 'already_snake_case']

print("Testing multiple cases:")
for test in test_cases:
    once = to_snake(test)
    twice = to_snake(once)
    thrice = to_snake(twice)
    print(f"Input: '{test:20}' -> once: '{once:20}' -> twice: '{twice:20}' -> thrice: '{thrice:20}' | Idempotent: {once == twice}")

print("\nDetailed analysis of 'A0' case:")
field = 'A0'
print(f"Original: '{field}'")

# Apply once
once = to_snake(field)
print(f"After 1st application: '{once}'")

# Apply twice
twice = to_snake(once)
print(f"After 2nd application: '{twice}'")

# Apply thrice to check if it stabilizes
thrice = to_snake(twice)
print(f"After 3rd application: '{thrice}'")

print(f"\nDoes it stabilize after 2nd application? {twice == thrice}")