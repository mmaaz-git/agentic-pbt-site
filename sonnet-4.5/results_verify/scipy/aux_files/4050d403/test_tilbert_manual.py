import numpy as np
from scipy import fftpack

x = np.array([0.0, 1.0])
x = x - x.mean()

print(f"Input (zero-mean): {x}")
print(f"sum(x) = {np.sum(x)}")

h = 0.5
print(f"h parameter: {h}")

t = fftpack.tilbert(x, h=h)
print(f"\ntilbert(x, h={h}) = {t}")

it = fftpack.itilbert(t, h=h)
print(f"itilbert(tilbert(x), h={h}) = {it}")

print(f"\nExpected: {x}")
print(f"Actual: {it}")
print(f"Match: {np.allclose(it, x, rtol=1e-3, atol=1e-5)}")

print("\n" + "="*60)
print("For comparison, length-3 works correctly:")
x3 = np.array([0.0, 1.0, 2.0])
x3 = x3 - x3.mean()
print(f"x3 = {x3}, sum = {np.sum(x3)}")
t3 = fftpack.tilbert(x3, h=h)
it3 = fftpack.itilbert(t3, h=h)
print(f"itilbert(tilbert(x3), h={h}) = {it3}")
print(f"Match: {np.allclose(it3, x3, rtol=1e-3, atol=1e-5)}")