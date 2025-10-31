import pandas.arrays as pa
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings


@settings(max_examples=200)
@given(st.integers(min_value=1, max_value=20))
def test_all_na_any_should_be_na(n):
    arr = pa.BooleanArray(np.zeros(n, dtype='bool'),
                          np.ones(n, dtype='bool'))

    any_result = arr.any()
    assert pd.isna(any_result), f"any() on all-NA array should return NA per Kleene logic, but got {any_result}"


@settings(max_examples=200)
@given(st.integers(min_value=1, max_value=20))
def test_all_na_all_should_be_na(n):
    arr = pa.BooleanArray(np.zeros(n, dtype='bool'),
                          np.ones(n, dtype='bool'))

    all_result = arr.all()
    assert pd.isna(all_result), f"all() on all-NA array should return NA per Kleene logic, but got {all_result}"

if __name__ == "__main__":
    # Run tests
    print("Running hypothesis tests...")
    test_all_na_any_should_be_na()
    test_all_na_all_should_be_na()
    print("Tests completed")