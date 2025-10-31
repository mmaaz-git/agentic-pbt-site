import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.lists(st.integers(min_value=-1, max_value=2), min_size=5, max_size=20))
@settings(max_examples=100)
def test_categorical_null_preservation(codes):
    categories = ['a', 'b', 'c']
    df = pd.DataFrame({'cat': pd.Categorical.from_codes(codes, categories=categories)})

    interchange_obj = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(interchange_obj)

    assert result['cat'].isna().sum() == df['cat'].isna().sum(), \
        f"Null count mismatch: {result['cat'].isna().sum()} != {df['cat'].isna().sum()}"


if __name__ == "__main__":
    test_categorical_null_preservation()