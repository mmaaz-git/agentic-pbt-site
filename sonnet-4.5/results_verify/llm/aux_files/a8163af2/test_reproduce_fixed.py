#!/usr/bin/env python3
"""Reproduce the reported bug in llm.utils.truncate_string - FIXED VERSION"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

print("=== Reproducing the Bug Report Example ===")
text = "hello world"
max_length = 2

result = truncate_string(text, max_length=max_length)
print(f"Input: {repr(text)}, max_length={max_length}")
print(f"Result: {repr(result)}")
print(f"Result length: {len(result)}")
print(f"Expected max length: {max_length}")
if len(result) > max_length:
    print(f"⚠️  BUG CONFIRMED: Result length {len(result)} exceeds max_length {max_length}")
print()

print("=== Understanding the Bug ===")
print("The code does: text[: max_length - 3] + '...'")
print(f"For max_length={max_length}, this becomes: text[:{max_length - 3}] + '...'")
print(f"Which is: text[:{max_length - 3}] + '...'")
print(f"text[:{max_length - 3}] = {repr(text[:max_length - 3])}")
print(f"So the result is: {repr(text[:max_length - 3] + '...')}")
print()

print("=== Testing More Edge Cases ===")
test_cases = [
    ("hello world", 0),
    ("hello world", 1),
    ("hello world", 2),
    ("hello world", 3),
    ("hello world", 4),
    ("hello world", 5),
    ("x", 0),
    ("x", 1),
    ("x", 2),
]

for text, max_length in test_cases:
    result = truncate_string(text, max_length=max_length)
    violation = len(result) > max_length
    status = "✗ VIOLATION" if violation else "✓ OK"
    print(f"truncate_string({repr(text)}, {max_length}) -> {repr(result)} (len={len(result)}) {status}")

print("\n=== Analysis ===")
print("When max_length < 3:")
print("  - max_length=0: text[:-3] + '...' returns last 3 chars removed + '...'")
print("  - max_length=1: text[:-2] + '...' returns last 2 chars removed + '...'")
print("  - max_length=2: text[:-1] + '...' returns last 1 char removed + '...'")
print("All of these result in strings longer than max_length!")
print()
print("The function violates its contract for max_length < 3.")