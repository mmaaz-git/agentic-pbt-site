from hypothesis import given, strategies as st, settings
import pandas as pd

@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
@settings(max_examples=500)
def test_split_rsplit_api_parity(strings, sep):
    s = pd.Series(strings)

    split_literal = s.str.split(sep, regex=False)

    rsplit_literal = s.str.rsplit(sep, regex=False)

    for i in range(len(s)):
        if pd.notna(s.iloc[i]):
            assert len(split_literal.iloc[i]) == len(rsplit_literal.iloc[i])

if __name__ == "__main__":
    test_split_rsplit_api_parity()