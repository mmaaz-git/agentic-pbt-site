#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, example
from Cython.Plex.Regexps import chars_to_ranges


@given(st.text(min_size=1, max_size=100))
@example('00')  # Specific failing case
@example('aaa')  # Another failing case
@example('aabbcc')  # Another failing case
@settings(max_examples=1000)
def test_chars_to_ranges_coverage(s):
    ranges = chars_to_ranges(s)

    covered_chars = set()
    for i in range(0, len(ranges), 2):
        code1, code2 = ranges[i], ranges[i + 1]
        for code in range(code1, code2):
            covered_chars.add(chr(code))

    original_chars = set(s)
    try:
        assert covered_chars == original_chars, f"Mismatch for input '{s}': expected {original_chars}, got {covered_chars}"
        print(f"✓ Passed for '{s}' (length {len(s)})")
    except AssertionError as e:
        print(f"✗ Failed: {e}")
        return False
    return True

# Run the test
print("Running hypothesis test...")
print("=" * 50)
test_chars_to_ranges_coverage()
print("=" * 50)
print("Test completed")