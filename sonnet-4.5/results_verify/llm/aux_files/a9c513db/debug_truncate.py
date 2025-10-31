#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Debug the actual implementation behavior
print("Understanding the implementation:")
print("=" * 50)

text = "hello"
for max_length in range(1, 10):
    result = truncate_string(text, max_length)

    # Calculate what the slicing actually does
    if len(text) > max_length:
        slice_end = max_length - 3
        sliced_text = text[:slice_end]
        expected = sliced_text + "..."
    else:
        expected = text

    print(f"max_length={max_length}: text[:{max_length-3}]='{ text[:max_length-3]}' + '...' = '{result}' (len={len(result)})")
    print(f"  - Slice index: {max_length - 3}")
    print(f"  - Expected based on code: '{expected}'")
    print(f"  - Actual result: '{result}'")
    print(f"  - Violates constraint: {len(result) > max_length}")
    print()

print("\nChecking with empty strings and edge cases:")
print("=" * 50)

edge_cases = [
    ("", 1),
    ("", 0),
    ("a", 0),
    ("a", 1),
    ("ab", 0),
    ("ab", 1),
    ("ab", 2),
    ("abc", 3),
]

for text, max_len in edge_cases:
    try:
        result = truncate_string(text, max_len)
        print(f"truncate_string('{text}', {max_len}) = '{result}' (len={len(result)})")
    except Exception as e:
        print(f"truncate_string('{text}', {max_len}) raised: {e}")