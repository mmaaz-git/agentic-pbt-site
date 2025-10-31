from scipy.optimize import bisect, ridder, brenth, brentq


def h(x):
    return x - 2.5  # Root is at 2.5, not at boundaries

a = 0.0
b = 5.0

print("Testing normal case where root is NOT at boundary:")
print("=" * 50)
print(f"Function: f(x) = x - 2.5")
print(f"Interval: [{a}, {b}]")
print(f"Root should be at: 2.5")
print()

for method in [bisect, ridder, brenth, brentq]:
    root, info = method(h, a, b, full_output=True)
    print(f"{method.__name__}:")
    print(f"  root = {root}")
    print(f"  iterations = {info.iterations}")
    print(f"  function_calls = {info.function_calls}")
    print(f"  converged = {info.converged}")

    # Check if iterations is reasonable
    if 0 <= info.iterations <= 100:
        print(f"  ✓ iterations is reasonable")
    else:
        print(f"  ✗ iterations is UNREASONABLE (garbage value)")
    print()