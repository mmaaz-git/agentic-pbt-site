from hypothesis import given, strategies as st, settings
import pandas.core.arrays as arrays

@given(st.lists(st.one_of(st.text(min_size=0, max_size=5), st.none()), min_size=1, max_size=30))
@settings(max_examples=500)
def test_categorical_codes_bounds(values):
    cat = arrays.Categorical(values)

    assert len(cat) == len(values)
    assert len(cat.codes) == len(values)

    for i, code in enumerate(cat.codes):
        if values[i] is None:
            assert code == -1, f"NA should have code -1 at index {i}, got {code}"
        else:
            assert 0 <= code < len(cat.categories), f"Code {code} out of bounds at index {i}"
            assert cat.categories[code] == values[i], f"Code mapping wrong at {i}"

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    test_categorical_codes_bounds()
    print("Test passed!")