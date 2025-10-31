from hypothesis import given, strategies as st, settings
from pandas.api.types import union_categoricals
import pandas as pd

categorical_strategy = st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20).map(
    lambda x: pd.Categorical(x)
)

@given(st.lists(categorical_strategy, min_size=1, max_size=5))
@settings(max_examples=500)
def test_union_categoricals_preserves_all_categories(categoricals):
    result = union_categoricals(categoricals)

    all_input_categories = set()
    for cat in categoricals:
        all_input_categories.update(cat.categories)

    result_categories = set(result.categories)

    assert all_input_categories == result_categories, \
        f"Categories mismatch. Input: {all_input_categories}, Result: {result_categories}"

if __name__ == "__main__":
    # Run the test
    test_union_categoricals_preserves_all_categories()