import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import math
from hypothesis import given, strategies as st, settings
from scipy import special


@given(
    a=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)
)
@settings(max_examples=1000)
def test_betaincc_betainccinv_roundtrip(a, b, y):
    x = special.betainccinv(a, b, y)

    if not (0 <= x <= 1):
        return

    result = special.betaincc(a, b, x)

    assert math.isclose(result, y, rel_tol=1e-7, abs_tol=1e-7), \
        f"betaincc({a}, {b}, betainccinv({a}, {b}, {y})) = {result}, expected {y}"

if __name__ == "__main__":
    test_betaincc_betainccinv_roundtrip()
    print("Test passed!")