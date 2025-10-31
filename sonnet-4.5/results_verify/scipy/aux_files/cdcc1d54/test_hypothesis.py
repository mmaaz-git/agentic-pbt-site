#!/usr/bin/env python3
"""Run the hypothesis test from the bug report"""
from hypothesis import given, strategies as st, settings
from scipy import odr
import numpy as np

@settings(max_examples=100)
@given(
    init_val=st.integers(min_value=0, max_value=2),
    so_init_val=st.integers(min_value=0, max_value=2)
)
def test_set_iprint_no_crash(init_val, so_init_val):
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    data = odr.Data(x, y)
    model = odr.unilinear
    odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0], rptfile='test.txt')

    try:
        odr_obj.set_iprint(init=init_val, so_init=so_init_val)
        print(f"✓ init={init_val}, so_init={so_init_val}")
    except Exception as e:
        print(f"✗ init={init_val}, so_init={so_init_val}: {e}")
        raise

# Run the test
if __name__ == "__main__":
    test_set_iprint_no_crash()