import pandas as pd
import pandas.api.types as pat
from hypothesis import given, strategies as st, assume
import numpy as np


@given(st.lists(st.one_of(st.integers(), st.text(), st.none()), min_size=1, max_size=10))
def test_pandas_dtype_should_handle_series_consistently(lst):
    series = pd.Series(lst)

    if series.dtype.kind != 'O':
        assume(False)

    result1 = pat.pandas_dtype(series.dtype)
    result2 = pat.pandas_dtype(series)

    assert result1 == result2, (
        f"pandas_dtype should return the same result for series.dtype and series itself, "
        f"but got {result1} for series.dtype and error for series"
    )

if __name__ == "__main__":
    # Test with the specific failing input
    try:
        lst = [None]
        series = pd.Series(lst)

        if series.dtype.kind == 'O':
            result1 = pat.pandas_dtype(series.dtype)
            print(f"pat.pandas_dtype(series.dtype) = {result1}")

            try:
                result2 = pat.pandas_dtype(series)
                print(f"pat.pandas_dtype(series) = {result2}")

                if result1 == result2:
                    print("Test passed: Both calls returned the same result")
                else:
                    print(f"Test failed: Different results - {result1} vs {result2}")
            except Exception as e2:
                print(f"pat.pandas_dtype(series) raised: {type(e2).__name__}: {e2}")
        else:
            print(f"Series dtype is not object: {series.dtype}")
    except Exception as e:
        print(f"Test failed: {e}")