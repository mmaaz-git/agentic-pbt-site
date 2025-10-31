import numpy as np
import scipy.signal
import warnings

# First, let's run the simple reproduction case
print("=" * 60)
print("SIMPLE REPRODUCTION TEST")
print("=" * 60)

b = np.array([0.0, 1.0])
a = np.array([1.0])

print("Original filter:")
print(f"  b = {b}, a = {a}")

impulse = np.array([1, 0, 0, 0, 0])
y_original = scipy.signal.lfilter(b, a, impulse)
print(f"  Impulse response: {y_original}")
print(f"  This represents H(z) = z^(-1), a delay of 1 sample")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    b_norm, a_norm = scipy.signal.normalize(b, a)
    if w:
        print(f"\nWarning generated: {w[0].message}")
        print(f"Warning category: {w[0].category.__name__}")

print(f"\nAfter normalize():")
print(f"  b = {b_norm}, a = {a_norm}")

y_normalized = scipy.signal.lfilter(b_norm, a_norm, impulse)
print(f"  Impulse response: {y_normalized}")
print(f"  This represents H(z) = 1, no delay")

print(f"\nFilter behavior preserved: {np.allclose(y_original, y_normalized)}")

# Now let's test the round-trip conversion issues
print("\n" + "=" * 60)
print("ROUND-TRIP CONVERSION TEST (tf -> zpk -> tf)")
print("=" * 60)

b_orig = np.array([0.0, 1.0])
a_orig = np.array([1.0])

print(f"Original transfer function: b={b_orig}, a={a_orig}")

# Convert to zpk
z, p, k = scipy.signal.tf2zpk(b_orig, a_orig)
print(f"ZPK representation: z={z}, p={p}, k={k}")

# Convert back to tf
b_recon, a_recon = scipy.signal.zpk2tf(z, p, k)
print(f"Reconstructed transfer function: b={b_recon}, a={a_recon}")

# Test impulse responses
y_orig = scipy.signal.lfilter(b_orig, a_orig, impulse)
y_recon = scipy.signal.lfilter(b_recon, a_recon, impulse)

print(f"\nOriginal impulse response: {y_orig}")
print(f"Reconstructed impulse response: {y_recon}")
print(f"Round-trip preserved: {np.allclose(y_orig, y_recon)}")

# Let's also test with a more complex example with leading zeros
print("\n" + "=" * 60)
print("TEST WITH MORE LEADING ZEROS")
print("=" * 60)

b_test = np.array([0.0, 0.0, 0.0, 2.5])
a_test = np.array([1.0])

print(f"Original filter: b={b_test}, a={a_test}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    b_norm_test, a_norm_test = scipy.signal.normalize(b_test, a_test)
    if w:
        print(f"Warning generated: {w[0].message}")

print(f"After normalize: b={b_norm_test}, a={a_norm_test}")

# Test impulse responses
impulse_long = np.array([1, 0, 0, 0, 0, 0, 0, 0])
y_orig_test = scipy.signal.lfilter(b_test, a_test, impulse_long)
y_norm_test = scipy.signal.lfilter(b_norm_test, a_norm_test, impulse_long)

print(f"\nOriginal impulse response: {y_orig_test}")
print(f"Normalized impulse response: {y_norm_test}")
print(f"Behavior preserved: {np.allclose(y_orig_test, y_norm_test)}")
print(f"This shows a delay of 3 samples becomes no delay!")

# Finally, run the property-based test with the specific failing case
print("\n" + "=" * 60)
print("PROPERTY-BASED TEST WITH FAILING INPUT")
print("=" * 60)

def test_tf2zpk_zpk2tf_roundtrip(b_coeffs, a_coeffs):
    b = np.array(b_coeffs)
    a = np.array(a_coeffs)

    if abs(a[0]) <= 1e-10:
        print("Skipping test due to near-zero a[0]")
        return True

    b = b / a[0]
    a = a / a[0]

    z, p, k = scipy.signal.tf2zpk(b, a)
    b_reconstructed, a_reconstructed = scipy.signal.zpk2tf(z, p, k)

    # Check that filter behavior is preserved
    impulse = np.array([1, 0, 0, 0, 0])
    y_orig = scipy.signal.lfilter(b, a, impulse)
    y_recon = scipy.signal.lfilter(b_reconstructed, a_reconstructed, impulse)

    try:
        np.testing.assert_allclose(y_orig, y_recon, rtol=1e-8, atol=1e-10)
        return True
    except AssertionError as e:
        print(f"Test failed with error: {e}")
        return False

# Test with the failing input
result = test_tf2zpk_zpk2tf_roundtrip([0.0, 1.0], [1.0])
print(f"Test passed: {result}")