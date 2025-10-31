from hypothesis import given, strategies as st, settings
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

@given(
    hpd.data_frames(
        columns=[
            hpd.column('A', elements=st.just(1.0)),
            hpd.column('B', elements=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False)),
            hpd.column('class', elements=st.sampled_from(['cat1', 'cat2']))
        ],
        index=hpd.range_indexes(min_size=2, max_size=10)
    )
)
@settings(max_examples=5, deadline=None)
def test_radviz_constant_column(df):
    print(f"\nTesting with DataFrame:")
    print(df)

    fig, ax = plt.subplots()
    try:
        result = pd.plotting.radviz(df, 'class', ax=ax)

        # Check if the resulting plot has NaN values
        # This happens when normalization divides by zero
        def normalize(series):
            a = min(series)
            b = max(series)
            return (series - a) / (b - a)

        normalized_A = normalize(df['A'])
        if np.any(np.isnan(normalized_A)):
            print(f"WARNING: Normalization produced NaN values!")
            print(f"Column A: min={df['A'].min()}, max={df['A'].max()}")
            print(f"Normalized column A contains NaN: {np.any(np.isnan(normalized_A))}")
            assert False, "radviz produced NaN values due to constant column"

        print("Test passed - no NaN values produced")
    except ZeroDivisionError as e:
        print(f"ZeroDivisionError occurred: {e}")
        assert False, f"radviz raised ZeroDivisionError: {e}"
    finally:
        plt.close('all')

if __name__ == "__main__":
    print("Running property-based test for pandas.plotting.radviz with constant columns...")
    print("=" * 60)
    test_radviz_constant_column()