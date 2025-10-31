import pandas as pd
import pandas.plotting
import matplotlib
matplotlib.use('Agg')
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.pandas import data_frames, columns, range_indexes

@settings(max_examples=200)
@given(
    df=data_frames(
        columns=columns(['a', 'b', 'c'], dtype=float),
        index=range_indexes(min_size=2, max_size=20)
    ),
    diagonal=st.text(min_size=1, max_size=10).filter(lambda x: x not in ['hist', 'kde'])
)
def test_scatter_matrix_diagonal_validation(df, diagonal):
    """
    Property: scatter_matrix should only accept 'hist' or 'kde' for diagonal parameter.
    The docstring explicitly states: diagonal : {'hist', 'kde'}
    """
    assume(len(df) >= 2)
    assume(len(df.columns) >= 2)

    try:
        result = pandas.plotting.scatter_matrix(df, diagonal=diagonal)
        assert False, f"scatter_matrix should reject diagonal='{diagonal}', but it didn't"
    except (ValueError, KeyError) as e:
        pass

if __name__ == "__main__":
    print("Running hypothesis test...")
    test_scatter_matrix_diagonal_validation()