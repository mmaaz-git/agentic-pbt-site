from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns
import pandas as pd
from dask.dataframe.dask_expr import from_pandas
from dask.dataframe.dask_expr.io.bag import to_bag
import dask

# Use synchronous scheduler to avoid multiprocessing issues
dask.config.set(scheduler='synchronous')

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

    assert len(result) == npartitions, f"Expected {npartitions} items, got {len(result)}"
    assert all(isinstance(item, pd.DataFrame) for item in result), f"Expected all items to be DataFrames, got types: {[type(item) for item in result]}"

if __name__ == '__main__':
    test_to_bag_frame_format_should_preserve_dataframes()