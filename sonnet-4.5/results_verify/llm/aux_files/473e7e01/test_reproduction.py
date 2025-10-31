#!/usr/bin/env python3
"""Reproduce the truncate_string bug report"""

import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from llm.utils import truncate_string

# Test the specific failing case mentioned in the bug report
print("Testing the specific failing case from bug report:")
print("-" * 50)
result = truncate_string('00', max_length=1)
print(f"Input: text='00', max_length=1")
print(f"Result: '{result}'")
print(f"Result length: {len(result)}")
print(f"Expected max length: 1")
print(f"Violates constraint: {len(result) > 1}")
print()

# Test a few more edge cases
print("Testing additional edge cases:")
print("-" * 50)

test_cases = [
    ('00', 1),
    ('abc', 2),
    ('hello', 3),
    ('test', 0),  # Edge case with 0
    ('', 1),      # Empty string
    ('x', 1),     # String exactly at limit
    ('xy', 1),    # String longer than limit
]

for text, max_length in test_cases:
    try:
        result = truncate_string(text, max_length=max_length)
        print(f"text='{text}', max_length={max_length} -> result='{result}', len={len(result)}, violates={len(result) > max_length}")
    except Exception as e:
        print(f"text='{text}', max_length={max_length} -> ERROR: {e}")

print()
print("Testing with Hypothesis property-based test:")
print("-" * 50)

# Run the hypothesis test
from hypothesis import given, strategies as st, settings
from hypothesis.database import ExampleDatabase

# Clear the hypothesis database to ensure fresh run
import tempfile
import os
tmpdir = tempfile.mkdtemp()

@given(st.text(), st.integers(min_value=1, max_value=1000))
@settings(max_examples=100, database=ExampleDatabase(tmpdir))
def test_truncate_string_length_bound(text, max_length):
    result = truncate_string(text, max_length=max_length)
    assert len(result) <= max_length, f"Result '{result}' (len={len(result)}) exceeds max_length={max_length} for input '{text}'"

try:
    test_truncate_string_length_bound()
    print("Hypothesis test PASSED for 100 examples")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")
except Exception as e:
    print(f"Hypothesis test ERROR: {e}")