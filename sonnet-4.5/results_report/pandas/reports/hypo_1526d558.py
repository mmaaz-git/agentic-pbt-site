import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.api.interchange import from_dataframe

@given(st.data())
@settings(max_examples=200)
def test_categorical_with_nulls(data):
    nrows = data.draw(st.integers(min_value=1, max_value=20))
    categories = data.draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
        min_size=1, max_size=5, unique=True
    ))

    codes = data.draw(st.lists(
        st.one_of(
            st.just(-1),
            st.integers(min_value=0, max_value=len(categories) - 1)
        ),
        min_size=nrows, max_size=nrows
    ))

    cat = pd.Categorical.from_codes(codes, categories=categories)
    df = pd.DataFrame({'col': cat})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    for i in range(len(result)):
        if pd.isna(df['col'].iloc[i]):
            assert pd.isna(result['col'].iloc[i]), f"Null at index {i} not preserved. Original: {df['col'].iloc[i]}, Result: {result['col'].iloc[i]}"
        else:
            assert df['col'].iloc[i] == result['col'].iloc[i], f"Value mismatch at index {i}. Original: {df['col'].iloc[i]}, Result: {result['col'].iloc[i]}"

if __name__ == "__main__":
    test_categorical_with_nulls()