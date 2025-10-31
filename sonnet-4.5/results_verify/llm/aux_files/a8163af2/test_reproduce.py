#!/usr/bin/env python3
"""Reproduce the reported bug in llm.utils.truncate_string"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import llm.utils as utils

# First, let's run the hypothesis test
print("=== Running Hypothesis Test ===")
@given(st.text(min_size=1), st.integers(min_value=0, max_value=100))
def test_truncate_string_respects_max_length(text, max_length):
    result = utils.truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result length {len(result)} exceeds max_length {max_length}. Text: {repr(text)}, Result: {repr(result)}"

try:
    test_truncate_string_respects_max_length()
    print("Hypothesis test passed (no failures found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Error running hypothesis test: {e}")

print("\n=== Reproducing Specific Examples ===")

# Test the specific example from the bug report
test_cases = [
    ("hello world", 2),
    ("hello", 2),
    ("test", 1),
    ("a", 0),
    ("longer text here", 3),
    ("x", 3),
    ("test string", 4),
    ("test string", 5),
    ("test string", 6),
]

for text, max_length in test_cases:
    result = utils.truncate_string(text, max_length=max_length)
    print(f"Input: {repr(text)}, max_length={max_length}")
    print(f"  Result: {repr(result)}")
    print(f"  Result length: {len(result)}")
    if len(result) > max_length:
        print(f"  ⚠️  VIOLATION: Result length {len(result)} exceeds max_length {max_length}")
    print()

print("\n=== Testing Edge Cases ===")

# Test with keep_end=True
print("With keep_end=True:")
for text, max_length in [("hello world", 8), ("hello world", 7), ("hello world", 6)]:
    result = utils.truncate_string(text, max_length=max_length, keep_end=True)
    print(f"Input: {repr(text)}, max_length={max_length}, keep_end=True")
    print(f"  Result: {repr(result)}")
    print(f"  Result length: {len(result)}")
    if len(result) > max_length:
        print(f"  ⚠️  VIOLATION: Result length {len(result)} exceeds max_length {max_length}")
    print()

# Test with normalize_whitespace=True
print("With normalize_whitespace=True:")
text = "hello    world    test"
for max_length in [10, 5, 2]:
    result = utils.truncate_string(text, max_length=max_length, normalize_whitespace=True)
    print(f"Input: {repr(text)}, max_length={max_length}, normalize_whitespace=True")
    print(f"  Result: {repr(result)}")
    print(f"  Result length: {len(result)}")
    if len(result) > max_length:
        print(f"  ⚠️  VIOLATION: Result length {len(result)} exceeds max_length {max_length}")
    print()

print("\n=== Analyzing the Implementation ===")
print("Looking at the source code:")
print("When max_length < 3 and text needs truncation:")
print("  The function returns: text[: max_length - 3] + '...'")
print("  For max_length=2: text[: 2-3] + '...' = text[:-1] + '...'")
print("  This always returns at least '...' (3 characters)")
print("  This violates the max_length constraint!")