from hypothesis import given, strategies as st, settings, example
from pandas.api.types import union_categoricals
import pandas as pd

categorical_strategy = st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=20).map(
    lambda x: pd.Categorical(x)
)

@given(st.lists(categorical_strategy, min_size=1, max_size=5))
@settings(max_examples=500)
@example([pd.Categorical(['0']), pd.Categorical(['0', '1\x00']), pd.Categorical(['1'])])
def test_union_categoricals_preserves_all_categories(categoricals):
    result = union_categoricals(categoricals)

    all_input_categories = set()
    for cat in categoricals:
        all_input_categories.update(cat.categories)

    result_categories = set(result.categories)

    if all_input_categories != result_categories:
        print(f"\nFailing case found!")
        print(f"Input categoricals:")
        for i, cat in enumerate(categoricals):
            print(f"  cat{i}: values={cat.tolist()}, categories={cat.categories.tolist()}")
        print(f"Expected categories: {sorted(all_input_categories)}")
        print(f"Actual categories: {sorted(result_categories)}")
        print(f"Missing categories: {all_input_categories - result_categories}")

    assert all_input_categories == result_categories, \
        f"Categories mismatch. Input: {all_input_categories}, Result: {result_categories}"

if __name__ == "__main__":
    print("Running hypothesis test...")
    try:
        test_union_categoricals_preserves_all_categories()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")