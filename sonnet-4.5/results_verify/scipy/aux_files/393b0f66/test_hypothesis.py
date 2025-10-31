from hypothesis import given, strategies as st, settings
import numpy as np

@given(
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=9)
)
@settings(max_examples=100)
def test_set_iprint_doesnt_crash(init, so_init, iter, final, iter_step):
    from scipy.odr import Data, Model, ODR

    def fcn(beta, x):
        return beta[0] * x + beta[1]

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    data = Data(x, y)
    model = Model(fcn)

    odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile='test.rpt')
    try:
        odr_obj.set_iprint(init=init, so_init=so_init, iter=iter, final=final, iter_step=iter_step)
        print(f"Success: init={init}, so_init={so_init}, iter={iter}, final={final}, iter_step={iter_step}")
    except Exception as e:
        print(f"FAILED: init={init}, so_init={so_init}, iter={iter}, final={final}, iter_step={iter_step}")
        print(f"  Error: {e}")
        raise

# Run the test
test_set_iprint_doesnt_crash()