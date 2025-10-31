#!/usr/bin/env python3
"""Test with hypothesis as described in bug report"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env')

import math
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from scipy.integrate import tanhsinh

@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e3, max_value=1e3),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
)
@settings(max_examples=10)  # Reduced for quick testing
def test_tanhsinh_constant(c, a, b):
    assume(abs(b - a) > 0.01)
    assume(abs(a) < 100 and abs(b) < 100)

    def f(x):
        return c

    try:
        result = tanhsinh(f, a, b)
        expected = c * (b - a)

        print(f"Testing c={c:.3f}, a={a:.3f}, b={b:.3f}")
        print(f"  Expected: {expected:.6f}")
        print(f"  Got: {result.integral:.6f}")

        assert math.isclose(result.integral, expected, rel_tol=1e-8, abs_tol=1e-10)
        print("  PASS")
    except IndexError as e:
        print(f"Testing c={c:.3f}, a={a:.3f}, b={b:.3f}")
        print(f"  FAIL: IndexError: {e}")
        raise
    except Exception as e:
        print(f"Testing c={c:.3f}, a={a:.3f}, b={b:.3f}")
        print(f"  FAIL: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    print("Running hypothesis test for tanhsinh with constant functions")
    print("=" * 60)
    test_tanhsinh_constant()