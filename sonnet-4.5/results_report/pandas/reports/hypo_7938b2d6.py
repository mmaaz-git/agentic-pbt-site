import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from pandas.core.interchange.from_dataframe import from_dataframe


@given(
    categories=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=10, unique=True),
    values_with_nulls=st.lists(
        st.one_of(st.integers(min_value=0, max_value=9), st.none()),
        min_size=1,
        max_size=20
    )
)
@settings(max_examples=300)
def test_categorical_nulls_preserved(categories, values_with_nulls):
    assume(len(categories) > 0)

    cat_values = []
    for v in values_with_nulls:
        if v is None:
            cat_values.append(None)
        else:
            cat_values.append(categories[v % len(categories)])

    try:
        cat_data = pd.Categorical(cat_values, categories=categories)
        df = pd.DataFrame({'cat': cat_data})

        original_null_count = df['cat'].isna().sum()

        result = from_dataframe(df.__dataframe__())

        result_null_count = result['cat'].isna().sum()

        if original_null_count != result_null_count:
            raise AssertionError(
                f"Null count changed! Original: {original_null_count}, "
                f"Result: {result_null_count}"
            )
    except Exception as e:
        if "Null count changed" in str(e):
            raise
        pass

if __name__ == "__main__":
    test_categorical_nulls_preserved()