#!/usr/bin/env python3

import sys
sys.path.append('/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

# First, let's test the hypothesis test case
from hypothesis import given, strategies as st
from llm.utils import truncate_string

@given(st.text(), st.integers(min_value=1, max_value=1000))
def test_truncate_string_length(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result '{result}' has length {len(result)} but max_length was {max_length}"

# Run the hypothesis test
print("Running hypothesis test...")
try:
    test_truncate_string_length()
    print("Hypothesis test passed")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now test the specific failing case mentioned in the bug report
print("\n" + "="*50)
print("Testing specific case: text='hello world', max_length=1")
print("="*50)

result = truncate_string("hello world", max_length=1)
print(f"Result: '{result}'")
print(f"Result length: {len(result)}")
print(f"Expected max length: 1")

if len(result) <= 1:
    print("✓ Test passed")
else:
    print("✗ Test failed - result exceeds max_length")

# Test with max_length=2
print("\n" + "="*50)
print("Testing specific case: text='hello world', max_length=2")
print("="*50)

result = truncate_string("hello world", max_length=2)
print(f"Result: '{result}'")
print(f"Result length: {len(result)}")
print(f"Expected max length: 2")

if len(result) <= 2:
    print("✓ Test passed")
else:
    print("✗ Test failed - result exceeds max_length")

# Test edge cases
print("\n" + "="*50)
print("Testing edge cases")
print("="*50)

test_cases = [
    ("hello", 0),
    ("hello", 1),
    ("hello", 2),
    ("hello", 3),
    ("hello", 4),
    ("hello", 5),
    ("hello", 6),
    ("hello world this is a test", 1),
    ("hello world this is a test", 2),
    ("hello world this is a test", 3),
    ("hello world this is a test", 10),
]

for text, max_length in test_cases:
    result = truncate_string(text, max_length)
    status = "✓" if len(result) <= max_length else "✗"
    print(f"{status} truncate_string('{text}', {max_length}) = '{result}' (len={len(result)})")

# Let's also test what the actual slicing logic does
print("\n" + "="*50)
print("Understanding the bug: what happens with negative slicing")
print("="*50)

text = "hello world"
for max_length in [1, 2, 3]:
    slice_index = max_length - 3
    sliced = text[:slice_index]
    result = sliced + "..."
    print(f"max_length={max_length}: text[:{slice_index}] + '...' = '{sliced}' + '...' = '{result}' (len={len(result)})")
    print(f"  - When max_length < 3, we slice with negative index which gives us more characters than expected")