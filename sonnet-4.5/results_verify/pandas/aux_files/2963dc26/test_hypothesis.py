from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.interchange import from_dataframe


@given(
    st.lists(st.sampled_from(['a', 'b', 'c']), min_size=5, max_size=20),
    st.integers(min_value=0, max_value=4)
)
def test_categorical_preserves_missing(categories, null_idx):
    codes = [0, 1, 2, -1, 0]
    cat = pd.Categorical.from_codes(codes, categories=['a', 'b', 'c'])
    df = pd.DataFrame({'cat': cat})

    result = from_dataframe(df.__dataframe__())

    assert result.isna().sum().sum() == df.isna().sum().sum()

if __name__ == "__main__":
    test_categorical_preserves_missing()