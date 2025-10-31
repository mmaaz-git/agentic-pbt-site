#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string

# Test the specific failing case from the bug report
text = "hello"
max_length = 2
result = truncate_string(text, max_length)

print(f"Input: text={repr(text)}, max_length={max_length}")
print(f"Output: {repr(result)}")
print(f"Length: {len(result)}")
print(f"Expected max length: {max_length}")
print(f"Violation: {len(result) > max_length}")

# Test additional edge cases
print("\n--- Testing more edge cases ---")
test_cases = [
    ("hello", 0),
    ("hello", 1),
    ("hello", 2),
    ("hello", 3),
    ("hello", 4),
    ("hello", 5),
    ("a", 1),
    ("ab", 1),
    ("abc", 1),
    ("abcd", 2),
]

for text, max_len in test_cases:
    result = truncate_string(text, max_len)
    violation = len(result) > max_len
    print(f"text={repr(text):8s}, max_length={max_len}, result={repr(result):10s}, len={len(result)}, violation={violation}")

# Try to understand what's happening with negative slicing
print("\n--- Understanding the bug mechanism ---")
text = "hello"
for max_length in range(0, 6):
    slice_index = max_length - 3
    sliced = text[:slice_index] if slice_index >= 0 else text[:slice_index]
    with_ellipsis = sliced + "..."
    print(f"max_length={max_length}, slice_index={slice_index}, text[:{slice_index}]={repr(sliced)}, result={repr(with_ellipsis)}")