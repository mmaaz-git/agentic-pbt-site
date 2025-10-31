from scipy.optimize import newton, bisect

def f(x):
    return x**2 - 4

def fprime(x):
    return 2 * x

print("Testing bisect with negative rtol:")
try:
    root = bisect(f, 0.0, 3.0, rtol=-0.1, disp=False)
    print(f"bisect accepted rtol=-0.1")
except ValueError as e:
    print(f"bisect rejected rtol=-0.1: {e}")

print("\nTesting newton with negative rtol:")
try:
    root = newton(f, 1.0, fprime=fprime, rtol=-0.1, disp=False)
    print(f"newton accepted rtol=-0.1, returned {root}")
except ValueError as e:
    print(f"newton rejected rtol=-0.1: {e}")