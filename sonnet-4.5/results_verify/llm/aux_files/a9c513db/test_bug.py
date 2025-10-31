#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.utils import truncate_string

print("Testing the Hypothesis property-based test:")
print("=" * 50)

# Run the property-based test
@given(
    st.text(min_size=1, max_size=1000),
    st.integers(min_value=1, max_value=500)
)
def test_truncate_string_length_constraint(text, max_length):
    result = truncate_string(text, max_length)
    assert len(result) <= max_length, f"Result length {len(result)} > max_length {max_length}"

# Try to run the test
try:
    test_truncate_string_length_constraint()
    print("Property-based test PASSED")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")
except Exception as e:
    print(f"Property-based test ERROR: {e}")

print("\n" + "=" * 50)
print("Testing specific failing example from bug report:")
print("=" * 50)

# Test the specific failing case
text = "hello"
max_length = 1
result = truncate_string(text, max_length)

print(f"Input: '{text}' (length {len(text)})")
print(f"Max length: {max_length}")
print(f"Result: '{result}' (length {len(result)})")
print(f"Constraint violated: {len(result) > max_length}")

print("\n" + "=" * 50)
print("Testing additional edge cases:")
print("=" * 50)

# Test more edge cases
test_cases = [
    ("hello", 1),
    ("hello", 2),
    ("hello", 3),
    ("hello", 4),
    ("hello", 5),
    ("hello", 6),
    ("a", 1),
    ("ab", 1),
    ("abc", 1),
    ("abcd", 2),
    ("abcde", 3),
]

for text, max_len in test_cases:
    result = truncate_string(text, max_len)
    violated = len(result) > max_len
    print(f"truncate_string('{text}', {max_len}) = '{result}' (len={len(result)}), violated={violated}")