import numpy as np
from scipy import fftpack

print("Testing the documented behavior:\n")

# Test with odd length (should work according to docs)
x_odd = np.array([-1.0, 0.0, 1.0])
print(f"Odd-length array: {x_odd}")
print(f"Sum: {np.sum(x_odd)}")
print(f"Length: {len(x_odd)} (odd)")

h = 0.5
t_odd = fftpack.tilbert(x_odd, h=h)
it_odd = fftpack.itilbert(t_odd, h=h)
print(f"After round-trip: {it_odd}")
print(f"Matches original: {np.allclose(it_odd, x_odd, rtol=1e-3, atol=1e-5)}")

print("\n" + "="*60)

# Test with even length (NOT guaranteed to work according to docs)
x_even = np.array([-0.5, 0.5])
print(f"Even-length array: {x_even}")
print(f"Sum: {np.sum(x_even)}")
print(f"Length: {len(x_even)} (even)")

t_even = fftpack.tilbert(x_even, h=h)
it_even = fftpack.itilbert(t_even, h=h)
print(f"After round-trip: {it_even}")
print(f"Matches original: {np.allclose(it_even, x_even, rtol=1e-3, atol=1e-5)}")

print("\n" + "="*60)
print("\nDocumentation says: 'If sum(x, axis=0) == 0 and n = len(x) is odd, then tilbert(itilbert(x)) == x'")
print("Notice: Round-trip is ONLY guaranteed for ODD-length sequences!")