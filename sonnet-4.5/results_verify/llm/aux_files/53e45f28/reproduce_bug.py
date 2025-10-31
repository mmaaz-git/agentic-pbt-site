#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

print("Testing the specific case from the bug report:")
print("=" * 50)

result = truncate_string("hello world", 1)
print(f"Input: truncate_string('hello world', 1)")
print(f"Result: '{result}'")
print(f"Length: {len(result)}")
print(f"Expected max: 1")
print()

print("Additional failing cases:")
print("=" * 50)

test_cases = [
    ("test", 1),
    ("test", 2),
    ("example", 1),
    ("example", 2),
    ("", 1),
    ("", 2),
    ("a", 1),
    ("ab", 2),
]

for text, max_length in test_cases:
    result = truncate_string(text, max_length)
    print(f"truncate_string('{text}', {max_length})")
    print(f"  Result: '{result}'")
    print(f"  Length: {len(result)} (expected max: {max_length})")
    print()