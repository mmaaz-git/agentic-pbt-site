import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings

@given(st.data())
@settings(max_examples=100)
def test_categorical_with_nulls_property(data):
    categories = data.draw(st.lists(
        st.text(min_size=1, max_size=5, alphabet=st.characters(min_codepoint=ord('a'), max_codepoint=ord('z'))),
        min_size=1,
        max_size=5,
        unique=True
    ))
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    values = data.draw(st.lists(
        st.one_of(
            st.sampled_from(categories),
            st.none()
        ),
        min_size=n_rows,
        max_size=n_rows
    ))

    df = pd.DataFrame({"cat": pd.Categorical(values, categories=categories)})

    null_count_before = df["cat"].isna().sum()

    df_interchange = df.__dataframe__()
    df_result = from_dataframe(df_interchange)

    null_count_after = df_result["cat"].isna().sum()

    assert null_count_before == null_count_after, f"Null count mismatch: {null_count_before} before, {null_count_after} after. Categories: {categories}, Values: {values}"

if __name__ == "__main__":
    test_categorical_with_nulls_property()