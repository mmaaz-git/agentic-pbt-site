from hypothesis import given, strategies as st, settings
import numpy as np
import scipy.linalg

@settings(max_examples=100)
@given(st.integers(min_value=2, max_value=30))
def test_pascal_invpascal_are_inverses(n):
    P = scipy.linalg.pascal(n)
    P_inv = scipy.linalg.invpascal(n, exact=False)

    I = np.eye(n)
    product1 = P @ P_inv
    product2 = P_inv @ P

    assert np.allclose(product1, I, rtol=1e-10, atol=1e-10), \
        f"pascal @ invpascal != I for n={n}, ||P @ P_inv - I|| = {np.linalg.norm(product1 - I)}"
    assert np.allclose(product2, I, rtol=1e-10, atol=1e-10), \
        f"invpascal @ pascal != I for n={n}, ||P_inv @ P - I|| = {np.linalg.norm(product2 - I)}"

if __name__ == "__main__":
    test_pascal_invpascal_are_inverses()