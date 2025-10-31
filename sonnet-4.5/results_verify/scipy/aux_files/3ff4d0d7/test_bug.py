from scipy.optimize import bisect, ridder, brenth, brentq


def f(x):
    return x - 5

a = 0.0
b = 5.0

print("Testing boundary root case where f(b) = 0:")
print("=" * 50)

for method in [bisect, ridder, brenth, brentq]:
    root, info = method(f, a, b, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print(f"  converged = {info.converged}")
    print()

print("\nTesting boundary root case where f(a) = 0:")
print("=" * 50)

def g(x):
    return x - 0

a2 = 0.0
b2 = 5.0

for method in [bisect, ridder, brenth, brentq]:
    root, info = method(g, a2, b2, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print(f"  converged = {info.converged}")
    print()