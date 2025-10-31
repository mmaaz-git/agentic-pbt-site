import numpy as np
import pandas as pd
from hypothesis import given, settings, strategies as st


@given(st.data())
@settings(max_examples=100)
def test_categorical_negative_sentinel_preserved(data):
    n_categories = data.draw(st.integers(min_value=2, max_value=10))
    categories = [f"cat_{i}" for i in range(n_categories)]
    n_rows = data.draw(st.integers(min_value=1, max_value=20))

    codes = []
    for _ in range(n_rows):
        is_null = data.draw(st.booleans())
        if is_null:
            codes.append(-1)
        else:
            codes.append(data.draw(st.integers(min_value=0, max_value=n_categories-1)))

    codes = np.array(codes, dtype=np.int64)
    cat_values = pd.Categorical.from_codes(codes, categories=categories)
    df = pd.DataFrame({"cat_col": cat_values})

    xchg = df.__dataframe__()
    result = pd.api.interchange.from_dataframe(xchg)

    np.testing.assert_array_equal(
        df["cat_col"].isna().values,
        result["cat_col"].isna().values,
        err_msg="Null positions don't match after interchange"
    )

if __name__ == "__main__":
    test_categorical_negative_sentinel_preserved()