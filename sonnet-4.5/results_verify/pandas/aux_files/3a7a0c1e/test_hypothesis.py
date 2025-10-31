#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings, example
import pandas as pd

@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
@example(['abc'], 2, 1)  # The specific failing case mentioned
@settings(max_examples=100)  # Reduced for faster testing
def test_slice_replace_consistency(strings, start, stop):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, 'X')

    for orig_str, repl in zip(strings, replaced):
        if start is None:
            actual_start = 0
        elif start < 0:
            actual_start = max(0, len(orig_str) + start)
        else:
            actual_start = start

        if stop is None:
            actual_stop = len(orig_str)
        elif stop < 0:
            actual_stop = max(0, len(orig_str) + stop)
        else:
            actual_stop = stop

        expected_repl = orig_str[:actual_start] + 'X' + orig_str[actual_stop:]

        if repl != expected_repl:
            print(f"\nFAILURE DETECTED:")
            print(f"  Original string: {orig_str!r}")
            print(f"  start={start}, stop={stop}")
            print(f"  actual_start={actual_start}, actual_stop={actual_stop}")
            print(f"  slice [{actual_start}:{actual_stop}] = {orig_str[actual_start:actual_stop]!r}")
            print(f"  Result:   {repl!r}")
            print(f"  Expected: {expected_repl!r}")
            raise AssertionError(f"Mismatch: {repl!r} != {expected_repl!r}")

# Run the test
if __name__ == "__main__":
    try:
        test_slice_replace_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")