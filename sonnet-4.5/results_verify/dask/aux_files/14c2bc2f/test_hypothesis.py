from hypothesis import given, settings, example
from hypothesis.extra.pandas import data_frames, column, range_indexes
import dask.dataframe as dd
import pandas as pd


@settings(max_examples=10)
@given(
    df=data_frames(
        columns=[
            column('x', dtype=int),
            column('y', dtype=int),
        ],
        index=range_indexes(min_size=1, max_size=50),
    ),
)
@example(pd.DataFrame({'x': [0, 0], 'y': [0, 0]}))
def test_reset_index_matches_pandas(df):
    ddf = dd.from_pandas(df, npartitions=2)

    dask_result = ddf.reset_index(drop=True).compute()
    pandas_result = df.reset_index(drop=True)

    pd.testing.assert_frame_equal(dask_result, pandas_result)


if __name__ == "__main__":
    # Test with the specific failing input mentioned in the bug report
    df = pd.DataFrame({'x': [0, 0], 'y': [0, 0]})
    print("Testing with specific failing input:")
    print(f"Input DataFrame:\n{df}")

    ddf = dd.from_pandas(df, npartitions=2)
    dask_result = ddf.reset_index(drop=True).compute()
    pandas_result = df.reset_index(drop=True)

    print(f"\nPandas result:\n{pandas_result}")
    print(f"Pandas index: {pandas_result.index.tolist()}")

    print(f"\nDask result:\n{dask_result}")
    print(f"Dask index: {dask_result.index.tolist()}")

    try:
        pd.testing.assert_frame_equal(dask_result, pandas_result)
        print("\nTest PASSED: Dask and Pandas results match")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")

    # Run the full hypothesis test
    print("\n" + "="*50)
    print("Running full hypothesis test...")
    try:
        test_reset_index_matches_pandas()
        print("All hypothesis tests passed!")
    except AssertionError as e:
        print(f"Hypothesis test failed: {e}")