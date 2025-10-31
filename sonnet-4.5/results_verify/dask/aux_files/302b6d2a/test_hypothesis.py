from hypothesis import given, strategies as st
from dask.diagnostics import ProgressBar
from dask.threaded import get
from operator import add
import io


@given(st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
def test_progress_bar_negative_dt(dt):
    output = io.StringIO()
    dsk = {'x': 1, 'y': (add, 'x', 10)}

    with ProgressBar(dt=dt, out=output):
        result = get(dsk, 'y')

    assert result == 11

if __name__ == "__main__":
    # Run the test
    test_progress_bar_negative_dt()