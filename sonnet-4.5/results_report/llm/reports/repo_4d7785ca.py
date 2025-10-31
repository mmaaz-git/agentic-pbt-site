#!/usr/bin/env python3
"""Minimal test case demonstrating truncate_string bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Test case 1: max_length = 1
print("Test 1: truncate_string('hello world', 1)")
result = truncate_string("hello world", 1)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 1")
print(f"  Violation: {len(result) > 1}")
print()

# Test case 2: max_length = 2
print("Test 2: truncate_string('hello world', 2)")
result = truncate_string("hello world", 2)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 2")
print(f"  Violation: {len(result) > 2}")
print()

# Test case 3: max_length = 3 (should work correctly)
print("Test 3: truncate_string('hello world', 3)")
result = truncate_string("hello world", 3)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 3")
print(f"  Violation: {len(result) > 3}")
print()

# Test case 4: Short string with small max_length
print("Test 4: truncate_string('ab', 1)")
result = truncate_string("ab", 1)
print(f"  Result: {repr(result)}")
print(f"  Length: {len(result)}")
print(f"  Expected max length: 1")
print(f"  Violation: {len(result) > 1}")