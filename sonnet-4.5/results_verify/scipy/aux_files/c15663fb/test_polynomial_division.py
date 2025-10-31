import numpy as np
from scipy import signal

print("Testing polynomial division and deconvolution semantic differences")
print("=" * 70)

# In polynomial representation, [0, 1] means the constant 1
# np.poly1d automatically strips leading zeros
p1 = np.poly1d([0, 1])
print(f"np.poly1d([0, 1]) gives polynomial: {p1}")
print(f"Its coefficients after normalization: {p1.coeffs}")
print()

# However, scipy.signal.deconvolve doesn't strip leading zeros
print("scipy.signal.deconvolve behavior:")
print("-" * 40)

# Case 1: Working case without leading zeros
print("Case 1: divisor=[1, 0] (polynomial x)")
signal_arr = np.array([1, 2, 3])
divisor = np.array([1, 0])
try:
    q, r = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={q}, remainder={r}")
    # Verify the relationship
    reconstructed = signal.convolve(divisor, q, mode='full') + r
    print(f"Reconstructed: {reconstructed[:len(signal_arr)]}")
    print(f"Original: {signal_arr}")
except Exception as e:
    print(f"Error: {e}")

print()

# Case 2: Failing case with leading zeros
print("Case 2: divisor=[0, 1] (polynomial constant 1)")
signal_arr = np.array([1, 2, 3])
divisor = np.array([0, 1])
try:
    q, r = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={q}, remainder={r}")
except Exception as e:
    print(f"Error: {e}")

print()

# Case 3: What if we manually strip the leading zero?
print("Case 3: Manually stripping leading zero from [0, 1] to get [1]")
signal_arr = np.array([1, 2, 3])
divisor = np.array([1])  # [0, 1] with leading zero stripped
try:
    q, r = signal.deconvolve(signal_arr, divisor)
    print(f"Success: quotient={q}, remainder={r}")
    # Verify the relationship
    reconstructed = signal.convolve(divisor, q, mode='full') + r
    print(f"Reconstructed: {reconstructed[:len(signal_arr)]}")
    print(f"Original: {signal_arr}")
    print("The division by constant 1 works correctly when stripped!")
except Exception as e:
    print(f"Error: {e}")

print()
print("Key observation:")
print("The array [0, 1] mathematically represents the constant polynomial 1.")
print("scipy.signal.deconvolve should handle this by stripping leading zeros,")
print("just like numpy.poly1d does automatically.")