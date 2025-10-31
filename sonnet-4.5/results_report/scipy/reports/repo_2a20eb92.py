import scipy.special as sp

a = 1e-05
y = 0.5

x = sp.gammainccinv(a, y)
print(f"gammainccinv({a}, {y}) = {x}")

result = sp.gammaincc(a, x)
print(f"gammaincc({a}, {x}) = {result}")
print(f"Expected: {y}")
print(f"Error: {abs(result - y)}")