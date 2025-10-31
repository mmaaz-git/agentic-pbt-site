import numpy as np
from hypothesis import given, strategies as st, settings
import scipy.signal as signal

# First, let's run the specific failing case mentioned
print("=== Testing specific failing case ===")
z = np.array([], dtype=complex)
p = np.array([-5.+0.j, -5.+0.j, -5.+0.j, -5.+0.j])
k = 1.0

print(f"Original zeros: {z}")
print(f"Original poles: {p}")
print(f"Original gain: {k}")

num, den = signal.zpk2tf(z, p, k)
print(f"\nTransfer function numerator: {num}")
print(f"Transfer function denominator: {den}")

z2, p2, k2 = signal.tf2zpk(num, den)
print(f"\nRecovered zeros: {z2}")
print(f"Recovered poles: {p2}")
print(f"Recovered gain: {k2}")

p_sorted = np.sort(p.real)
p2_sorted = np.sort(p2.real)

max_error = np.max(np.abs(p_sorted - p2_sorted))

print(f"\nOriginal poles (sorted): {p_sorted}")
print(f"Recovered poles (sorted): {p2_sorted}")
print(f"Maximum error: {max_error:.6e}")
print(f"Relative error: {max_error/5.0:.2%}")

# Now let's test the hypothesis property-based test
print("\n=== Running Hypothesis test ===")

@st.composite
def repeated_poles_zpk(draw):
    pole_value = draw(st.floats(min_value=-5, max_value=-0.1, allow_nan=False, allow_infinity=False))
    n_repeats = draw(st.integers(min_value=2, max_value=5))

    poles = np.full(n_repeats, pole_value, dtype=complex)
    zeros = np.array([], dtype=complex)
    gain = 1.0

    return zeros, poles, gain

failures = []

@settings(max_examples=50)
@given(repeated_poles_zpk())
def test_repeated_poles_roundtrip(zpk_data):
    z1, p1, k1 = zpk_data

    num, den = signal.zpk2tf(z1, p1, k1)
    z2, p2, k2 = signal.tf2zpk(num, den)

    p1_sorted = np.sort(p1.real)
    p2_sorted = np.sort(p2.real)

    max_error = np.max(np.abs(p1_sorted - p2_sorted))

    if max_error >= 1e-3:
        failures.append({
            'original_poles': p1_sorted,
            'recovered_poles': p2_sorted,
            'max_error': max_error,
            'pole_value': p1[0].real,
            'n_repeats': len(p1)
        })

try:
    test_repeated_poles_roundtrip()
    print(f"Test completed. Found {len(failures)} failures out of 50 examples.")
except:
    print(f"Test failed with assertion error. Found {len(failures)} failures.")

if failures:
    print(f"\n=== Sample failures ===")
    for i, fail in enumerate(failures[:3]):  # Show first 3 failures
        print(f"\nFailure {i+1}:")
        print(f"  Pole value: {fail['pole_value']}")
        print(f"  Number of repeats: {fail['n_repeats']}")
        print(f"  Max error: {fail['max_error']:.6e}")
        print(f"  Relative error: {fail['max_error']/abs(fail['pole_value']):.2%}")

# Test with different repeat counts
print("\n=== Testing error vs number of repeated poles ===")
for n_repeats in range(2, 6):
    p = np.array([-5.+0.j] * n_repeats)
    z = np.array([], dtype=complex)
    k = 1.0

    num, den = signal.zpk2tf(z, p, k)
    z2, p2, k2 = signal.tf2zpk(num, den)

    p_sorted = np.sort(p.real)
    p2_sorted = np.sort(p2.real)

    max_error = np.max(np.abs(p_sorted - p2_sorted))
    print(f"{n_repeats} repeated poles: max error = {max_error:.6e}, relative = {max_error/5.0:.2%}")

# Test with non-repeated poles
print("\n=== Testing with non-repeated poles ===")
p = np.array([-1.+0.j, -2.+0.j, -3.+0.j, -4.+0.j])
z = np.array([], dtype=complex)
k = 1.0

num, den = signal.zpk2tf(z, p, k)
z2, p2, k2 = signal.tf2zpk(num, den)

p_sorted = np.sort(p.real)
p2_sorted = np.sort(p2.real)

max_error = np.max(np.abs(p_sorted - p2_sorted))
print(f"Original poles: {p_sorted}")
print(f"Recovered poles: {p2_sorted}")
print(f"Maximum error: {max_error:.6e}")
print(f"Relative error (avg): {(max_error/np.mean(np.abs(p_sorted))):.2%}")