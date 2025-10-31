from hypothesis import given, strategies as st, settings
import pandas as pd

@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=20),
    st.integers(min_value=-10, max_value=10),
    st.integers(min_value=-10, max_value=10)
)
@settings(max_examples=1000)
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
        assert repl == expected_repl, f"Failed for {orig_str!r} with start={start}, stop={stop}. Got {repl!r}, expected {expected_repl!r}"

# Run the test
if __name__ == "__main__":
    test_slice_replace_consistency()