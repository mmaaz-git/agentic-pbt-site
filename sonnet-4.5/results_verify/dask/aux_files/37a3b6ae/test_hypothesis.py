import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
import dask.dataframe as dd


@given(st.integers(min_value=1, max_value=30))
@settings(max_examples=10)  # Reduced for quick testing
def test_reset_index_round_trip(n):
    df = pd.DataFrame({
        'a': np.random.randint(0, 100, n)
    })
    df.index = pd.Index(range(10, 10+n), name='idx')

    ddf = dd.from_pandas(df, npartitions=2)

    reset = ddf.reset_index()
    result = reset.compute()

    expected = df.reset_index()

    try:
        pd.testing.assert_frame_equal(result, expected)
        print(f"PASS for n={n}")
    except AssertionError as e:
        print(f"FAIL for n={n}")
        print(f"  Result index: {result.index.tolist()}")
        print(f"  Expected index: {expected.index.tolist()}")
        return False
    return True

# Run the test
test_reset_index_round_trip()