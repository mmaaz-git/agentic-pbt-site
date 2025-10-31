from hypothesis import given, settings, strategies as st
import pandas as pd


@given(st.lists(st.integers(min_value=-10, max_value=10) | st.none(), min_size=1, max_size=30))
@settings(max_examples=500)
def test_integerarray_one_pow_x_is_one(exponents):
    arr = pd.array(exponents, dtype="Int64")
    base = pd.array([1] * len(arr), dtype="Int64")
    result = base ** arr

    for i in range(len(result)):
        if pd.notna(result[i]):
            assert result[i] == 1

# Run the test
if __name__ == "__main__":
    test_integerarray_one_pow_x_is_one()