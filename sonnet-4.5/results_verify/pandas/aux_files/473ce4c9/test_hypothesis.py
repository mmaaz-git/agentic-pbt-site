from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from pandas.core.interchange.from_dataframe import from_dataframe


@settings(max_examples=500)
@given(
    n_categories=st.integers(min_value=1, max_value=10),
    codes=st.lists(
        st.one_of(
            st.integers(min_value=0, max_value=9),
            st.just(-1)
        ),
        min_size=1,
        max_size=50
    )
)
def test_categorical_roundtrip_preserves_nulls(n_categories, codes):
    categories = [f'cat{i}' for i in range(n_categories)]
    codes_arr = np.array([c if c < n_categories else -1 for c in codes], dtype=np.int8)

    cat = pd.Categorical.from_codes(codes_arr, categories=categories)
    df_original = pd.DataFrame({'col': cat})

    df_result = from_dataframe(df_original.__dataframe__())

    for i in range(len(df_original)):
        if pd.isna(df_original['col'].iloc[i]):
            assert pd.isna(df_result['col'].iloc[i]), \
                f"Expected NaN at index {i}, got '{df_result['col'].iloc[i]}'"

if __name__ == "__main__":
    test_categorical_roundtrip_preserves_nulls()
    print("Test completed")