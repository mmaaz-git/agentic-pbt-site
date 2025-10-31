import pandas as pd
from hypothesis import given, strategies as st, settings


@given(
    strings=st.lists(st.text(), min_size=1),
    start=st.integers(min_value=-5, max_value=5) | st.none(),
    stop=st.integers(min_value=-5, max_value=10) | st.none(),
    repl=st.text()
)
@settings(max_examples=500)
def test_slice_replace_property(strings, start, stop, repl):
    s = pd.Series(strings)
    result = s.str.slice_replace(start, stop, repl)

    for i, string in enumerate(strings):
        if pd.isna(string):
            assert pd.isna(result.iloc[i])
            continue

        expected = string[:start] + repl + string[stop:]
        actual = result.iloc[i]

        assert actual == expected

# Run the test
test_slice_replace_property()