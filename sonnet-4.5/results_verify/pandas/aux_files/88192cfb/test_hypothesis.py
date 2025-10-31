import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.text(min_size=1), min_size=1, max_size=100), st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10), st.text())
@settings(max_examples=1000)
def test_slice_replace_consistency(strings, start, stop, repl):
    s = pd.Series(strings)
    replaced = s.str.slice_replace(start, stop, repl)

    for i in range(len(s)):
        if pd.notna(s.iloc[i]) and pd.notna(replaced.iloc[i]):
            expected = s.iloc[i][:start] + repl + s.iloc[i][stop:]
            assert replaced.iloc[i] == expected

# Test with the specific failing input
if __name__ == "__main__":
    print("Testing with specific failing input: strings=['hello'], start=1, stop=0, repl='X'")
    test_slice_replace_consistency(['hello'], 1, 0, 'X')
    print("Test passed!")