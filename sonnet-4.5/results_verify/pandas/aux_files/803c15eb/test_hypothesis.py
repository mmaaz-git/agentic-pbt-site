import pandas as pd
from hypothesis import given, strategies as st
import pandas.util

@given(st.lists(st.integers(), min_size=1, max_size=100))
def test_hash_pandas_object_hash_key_parameter(lst):
    series = pd.Series(lst)
    hash1 = pandas.util.hash_pandas_object(series, hash_key="key1")
    hash2 = pandas.util.hash_pandas_object(series, hash_key="key2")

if __name__ == "__main__":
    # Run the hypothesis test
    test_hash_pandas_object_hash_key_parameter()