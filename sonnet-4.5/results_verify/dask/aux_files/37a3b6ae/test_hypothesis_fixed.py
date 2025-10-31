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

# Run the test
test_reset_index_round_trip()

# Specifically test n=2 as mentioned in the bug report
print("\n--- Testing n=2 specifically (mentioned in bug report) ---")
df = pd.DataFrame({
    'a': np.random.randint(0, 100, 2)
})
df.index = pd.Index(range(10, 12), name='idx')

ddf = dd.from_pandas(df, npartitions=2)
reset = ddf.reset_index()
result = reset.compute()
expected = df.reset_index()

print(f"n=2 case:")
print(f"  Result index: {result.index.tolist()}")
print(f"  Expected index: {expected.index.tolist()}")
print(f"  Match: {result.index.tolist() == expected.index.tolist()}")