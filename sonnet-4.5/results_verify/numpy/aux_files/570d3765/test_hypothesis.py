from hypothesis import given, strategies as st, settings
from scipy.odr import Data, Model, ODR
import numpy as np


@given(
    init=st.integers(min_value=-5, max_value=10),
    so_init=st.integers(min_value=-5, max_value=10),
)
@settings(max_examples=200)
def test_set_iprint_invalid_values(init, so_init):
    def fcn(B, x):
        return B[0] * x + B[1]

    model = Model(fcn)
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    data = Data(x, y)
    odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile="test.txt")

    ip2arg = [[0, 0], [1, 0], [2, 0], [1, 1], [2, 1], [1, 2], [2, 2]]

    expected_valid = [init, so_init] in ip2arg

    try:
        odr_obj.set_iprint(init=init, so_init=so_init)
        if not expected_valid:
            assert False, f"Should have raised ValueError for invalid combination [{init}, {so_init}]"
    except ValueError as e:
        if expected_valid:
            assert False, f"Should not have raised ValueError for valid combination [{init}, {so_init}]"
        if "[" not in str(e) and "not in list" not in str(e):
            assert False, f"ValueError message is unhelpful: {e}"


if __name__ == "__main__":
    test_set_iprint_invalid_values()
    print("Test completed")