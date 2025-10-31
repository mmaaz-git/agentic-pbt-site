from hypothesis import given, strategies as st, settings, assume
import numpy as np
import scipy.interpolate as si

@settings(max_examples=500)
@given(
    st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
             min_size=2, max_size=50).flatmap(
        lambda x_list: st.tuples(
            st.just(sorted(set(x_list))),
            st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                     min_size=len(set(x_list)), max_size=len(set(x_list)))
        )
    ).filter(lambda xy: len(xy[0]) >= 2)
)
def test_splrep_splev_round_trip(xy):
    x, y = xy
    x = np.array(x)
    y = np.array(y)

    assume(len(x) >= 2)
    assume(len(x) == len(y))
    assume(np.all(np.diff(x) > 0))

    tck = si.splrep(x, y, s=0)
    y_evaluated = si.splev(x, tck)

    assert np.allclose(y, y_evaluated, rtol=1e-9, atol=1e-9), \
        f"splev should return original y values at original x. Expected {y}, got {y_evaluated}"

if __name__ == "__main__":
    test_splrep_splev_round_trip()