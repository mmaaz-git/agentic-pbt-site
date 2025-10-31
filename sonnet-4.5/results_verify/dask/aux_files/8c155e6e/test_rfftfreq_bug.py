#!/usr/bin/env python3
"""Test to reproduce the rfftfreq negative n bug"""

from hypothesis import given, strategies as st
import numpy.fft as fft
import numpy as np

# Property-based test from bug report
@given(st.integers(min_value=-10, max_value=10))
def test_rfftfreq_zero_or_negative_n(n):
    if n <= 0:
        try:
            result = fft.rfftfreq(n)
            assert False, f"rfftfreq should reject n={n}"
        except (ValueError, ZeroDivisionError):
            pass
    else:
        result = fft.rfftfreq(n)
        assert len(result) == n // 2 + 1

# Direct reproduction test
def test_direct_reproduction():
    print("Testing rfftfreq with n=-1:")
    try:
        result = fft.rfftfreq(-1)
        print(f"  Result: {result}")
        print(f"  Type: {type(result)}")
        print(f"  Length: {len(result)}")
    except Exception as e:
        print(f"  Exception raised: {e}")

    print("\nTesting fftfreq with n=-1 for comparison:")
    try:
        result = fft.fftfreq(-1)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception raised: {e}")

    print("\nTesting rfft with n=-1 for comparison:")
    try:
        x = np.array([1, 2, 3, 4])
        result = fft.rfft(x, n=-1)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Exception raised: {e}")

if __name__ == "__main__":
    # Run direct reproduction first
    test_direct_reproduction()

    print("\n" + "="*50)
    print("Running property-based test...")

    # Run the property-based test
    try:
        test_rfftfreq_zero_or_negative_n()
        print("Property-based test completed without failures")
    except AssertionError as e:
        print(f"Property-based test failed: {e}")