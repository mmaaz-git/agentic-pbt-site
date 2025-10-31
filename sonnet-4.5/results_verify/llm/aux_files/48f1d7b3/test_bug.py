#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.utils import truncate_string
from hypothesis import given, strategies as st

# First, run the property-based test
@given(
    st.text(min_size=1, max_size=100),
    st.integers(min_value=0, max_value=10)
)
def test_truncate_string_length_invariant(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, \
        f"Result length {len(result)} exceeds max_length {max_length}: '{result}'"

# Test with the specific failing input
print("Testing with specific failing input from bug report:")
try:
    result = truncate_string("Hello world", max_length=2)
    assert len(result) <= 2, f"Result length {len(result)} exceeds max_length 2: '{result}'"
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Now reproduce the exact example from the bug report
print("\nReproducing the exact example from bug report:")
text = "Hello world"

for max_length in [0, 1, 2, 3, 4]:
    result = truncate_string(text, max_length=max_length)
    print(f"max_length={max_length}: '{result}' (actual length={len(result)})")

print("\nVerifying the bug exists:")
for max_length in [0, 1, 2]:
    result = truncate_string(text, max_length=max_length)
    if len(result) > max_length:
        print(f"BUG CONFIRMED: max_length={max_length}, but result length={len(result)}")