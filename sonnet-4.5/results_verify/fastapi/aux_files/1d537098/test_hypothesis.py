from hypothesis import given, strategies as st, settings
import pandas as pd
import numpy as np
from dask.dataframe.dask_expr import from_pandas


@given(
    n_rows=st.integers(min_value=5, max_value=1000),
    n_partitions_initial=st.integers(min_value=2, max_value=20),
    n_partitions_target=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=200)
def test_repartition_division_count(n_rows, n_partitions_initial, n_partitions_target):
    pdf = pd.DataFrame({
        'x': np.arange(n_rows)
    }, index=np.arange(n_rows))

    df = from_pandas(pdf, npartitions=n_partitions_initial)
    repartitioned = df.repartition(npartitions=n_partitions_target)

    expected_division_count = n_partitions_target + 1
    actual_division_count = len(repartitioned.divisions)

    assert actual_division_count == expected_division_count, \
        f"Division count incorrect: expected {expected_division_count}, got {actual_division_count}"

if __name__ == "__main__":
    # Run the test
    test_repartition_division_count()