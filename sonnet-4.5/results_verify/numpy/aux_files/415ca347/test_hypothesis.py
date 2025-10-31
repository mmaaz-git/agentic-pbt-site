import numpy as np
from hypothesis import given, settings, strategies as st


@settings(max_examples=1000)
@given(st.integers(min_value=1, max_value=10**10), st.integers(min_value=1, max_value=10**10))
def test_gcd_lcm_product(a, b):
    gcd_val = np.gcd(a, b)
    lcm_val = np.lcm(a, b)
    product = gcd_val * lcm_val
    expected = abs(a * b)
    assert product == expected

if __name__ == "__main__":
    # Run the test
    test_gcd_lcm_product()