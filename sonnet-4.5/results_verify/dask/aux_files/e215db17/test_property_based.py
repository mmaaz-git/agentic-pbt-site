from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
import pandas as pd
import dask
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag

# Use single-threaded scheduler to avoid multiprocessing issues
dask.config.set(scheduler='single-threaded')

@given(
    df=data_frames(
        columns=columns(['A', 'B'], dtype=float),
        rows=st.tuples(st.just(1), st.integers(min_value=2, max_value=10))
    ),
    npartitions=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=50, deadline=None)
def test_to_bag_frame_format_should_preserve_dataframes(df, npartitions):
    assume(len(df) >= npartitions)

    ddf = from_pandas(df, npartitions=npartitions)
    bag = to_bag(ddf, format='frame', index=False)
    result = bag.compute()

    print(f"Test case: df.shape={df.shape}, npartitions={npartitions}")
    print(f"  Expected: {npartitions} DataFrame objects")
    print(f"  Got: {len(result)} items of types {[type(r).__name__ for r in result]}")

    assert len(result) == npartitions, f"Expected {npartitions} items, got {len(result)}"
    assert all(isinstance(item, pd.DataFrame) for item in result), f"Expected DataFrames, got {[type(r).__name__ for r in result]}"

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test...")
    print("=" * 60)
    try:
        test_to_bag_frame_format_should_preserve_dataframes()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
    except Exception as e:
        print(f"Test error: {e}")