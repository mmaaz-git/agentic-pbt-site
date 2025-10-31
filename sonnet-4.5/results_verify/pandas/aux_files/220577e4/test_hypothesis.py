import pandas as pd
from hypothesis import given, strategies as st, settings, example


@given(
    st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
    st.integers(min_value=0, max_value=10),
    st.integers(min_value=0, max_value=10),
    st.text(max_size=10)
)
@settings(max_examples=500)
@example(strings=['0'], start=1, stop=0, repl='')  # The failing example from the report
def test_slice_replace_matches_python(strings, start, stop, repl):
    s = pd.Series(strings)
    pandas_result = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if isinstance(s.iloc[i], str):
            original = s.iloc[i]
            expected = original[:start] + repl + original[stop:]
            print(f"String: {original!r}, start={start}, stop={stop}, repl={repl!r}")
            print(f"  Pandas result: {pandas_result.iloc[i]!r}")
            print(f"  Expected:      {expected!r}")
            assert pandas_result.iloc[i] == expected, f"Mismatch for {original!r} with start={start}, stop={stop}"

if __name__ == "__main__":
    test_slice_replace_matches_python()