from scipy.optimize import bisect, ridder, brenth, brentq


def f(x):
    return x - 5

a = 0.0
b = 5.0

print("Testing boundary root (root at b=5):")
print("=" * 50)
for method in [bisect, ridder, brenth, brentq]:
    root, info = method(f, a, b, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print(f"  converged = {info.converged}")
    print()