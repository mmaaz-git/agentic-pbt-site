from hypothesis import given, strategies as st, settings
from hypothesis.extra import pandas as hpd
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@given(
    df=hpd.data_frames(
        columns=[
            hpd.column('A', elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
            hpd.column('B', elements=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        ],
        index=hpd.range_indexes(min_size=2, max_size=50)
    ),
    invalid_diagonal=st.text(min_size=1, max_size=10).filter(lambda x: x not in ['hist', 'kde', 'density'])
)
@settings(max_examples=50, deadline=None)
def test_scatter_matrix_invalid_diagonal(df, invalid_diagonal):
    try:
        result = pd.plotting.scatter_matrix(df, diagonal=invalid_diagonal)
        plt.close('all')
    except (ValueError, KeyError) as e:
        plt.close('all')
        return

    raise AssertionError(f"Expected error for invalid diagonal '{invalid_diagonal}', but got success")

if __name__ == "__main__":
    test_scatter_matrix_invalid_diagonal()
    print("Test completed")