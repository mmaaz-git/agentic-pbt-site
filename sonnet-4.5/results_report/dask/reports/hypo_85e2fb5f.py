import pandas as pd
import dask.dataframe as dd
import numpy as np
from hypothesis import given, strategies as st, settings
import traceback


@given(
    st.integers(min_value=2, max_value=10),
    st.sampled_from(['Q', 'QE', 'M', 'ME']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['left', 'right']),
    st.sampled_from(['sum', 'mean', 'count'])
)
@settings(max_examples=500, deadline=None)
def test_resample_matches_pandas(npartitions, rule, closed, label, method):
    start = pd.Timestamp('2020-01-01')
    end = pd.Timestamp('2020-12-31')

    dates = pd.date_range(start, end, freq='D')
    data = pd.DataFrame({'value': np.arange(len(dates), dtype=float)}, index=dates)

    ddf = dd.from_pandas(data, npartitions=npartitions)

    pandas_result = getattr(data.resample(rule, closed=closed, label=label), method)()
    dask_result = getattr(ddf.resample(rule, closed=closed, label=label), method)().compute()

    pd.testing.assert_frame_equal(
        pandas_result.sort_index(),
        dask_result.sort_index(),
        check_dtype=False,
        atol=1e-10
    )

# Test with the specific failing case
print("Testing with specific failing input: npartitions=4, rule='Q', closed='right', label='right', method='sum'")
try:
    test_resample_matches_pandas(npartitions=4, rule='Q', closed='right', label='right', method='sum')
    print("Test passed (unexpected!)")
except AssertionError as e:
    print("Test failed as expected with AssertionError:")
    print(str(e))
    print("\nFailing input: npartitions=4, rule='Q', closed='right', label='right', method='sum'")
except Exception as e:
    print(f"Test failed with {type(e).__name__}:")
    print(traceback.format_exc())