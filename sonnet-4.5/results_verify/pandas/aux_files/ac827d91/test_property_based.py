import pandas.core.arrays as arrays
from hypothesis import given, settings, strategies as st, assume

def categoricals():
    @st.composite
    def _categoricals(draw):
        categories = draw(st.lists(st.text(min_size=1, max_size=5), min_size=1, max_size=10, unique=True))
        codes = draw(st.lists(st.integers(min_value=-1, max_value=len(categories)-1), min_size=1, max_size=50))
        return arrays.Categorical.from_codes(codes, categories=categories)
    return _categoricals()

@given(categoricals())
@settings(max_examples=200)
def test_categorical_add_remove_categories_identity(cat):
    original_categories = list(cat.categories)
    new_cat = 'NEW_CATEGORY_XYZ'
    assume(new_cat not in original_categories)

    cat_added = cat.add_categories([new_cat])
    cat_removed = cat_added.remove_categories([new_cat])

    assert list(cat_removed.categories) == original_categories, f"Expected {original_categories}, got {list(cat_removed.categories)}"

if __name__ == "__main__":
    test_categorical_add_remove_categories_identity()
    print("Property-based test completed")