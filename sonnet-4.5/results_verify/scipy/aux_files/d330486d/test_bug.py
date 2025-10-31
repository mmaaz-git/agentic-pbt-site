from scipy.optimize.cython_optimize import _zeros
import scipy.optimize

a0 = -1.0
xa = 2.0
xb = 3.0

def f(x):
    return x**3 + a0

print("=" * 60)
print("Testing scipy.optimize root-finding with no sign change")
print("=" * 60)
print(f"\nFunction: f(x) = x^3 + {a0}")
print(f"Interval: [{xa}, {xb}]")
print(f"f(xa) = f({xa}) = {f(xa)}")
print(f"f(xb) = f({xb}) = {f(xb)}")
print(f"Both positive - no sign change!\n")

print("-" * 60)
print("scipy.optimize.bisect (Python API):")
print("-" * 60)
try:
    root = scipy.optimize.bisect(f, xa, xb)
    print(f"  Result: {root}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\n" + "-" * 60)
print("scipy.optimize.brentq (Python API):")
print("-" * 60)
try:
    root = scipy.optimize.brentq(f, xa, xb)
    print(f"  Result: {root}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\n" + "-" * 60)
print("scipy.optimize.brenth (Python API):")
print("-" * 60)
try:
    root = scipy.optimize.brenth(f, xa, xb)
    print(f"  Result: {root}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\n" + "-" * 60)
print("scipy.optimize.ridder (Python API):")
print("-" * 60)
try:
    root = scipy.optimize.ridder(f, xa, xb)
    print(f"  Result: {root}")
except ValueError as e:
    print(f"  Correctly raises ValueError: {e}")
except Exception as e:
    print(f"  Unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("_zeros.loop_example (Cython API):")
print("=" * 60)
for method in ['bisect', 'brentq', 'brenth', 'ridder']:
    print(f"\nMethod: {method}")
    try:
        results = list(_zeros.loop_example(method, (a0,), (0.0, 0.0, 1.0), xa, xb, 0.01, 0.01, 50))
        print(f"  Returns: {results}")
        if results:
            root = results[0]
            print(f"    root={root}")
            print(f"    Is root in [{xa}, {xb}]? {xa <= root <= xb}")
            print(f"    f(root)={f(root)} (should be ≈0)")
            # Check if it's actually a valid root
            if abs(f(root)) > 0.01:
                print(f"    WARNING: f(root) is not close to 0!")
            if not (xa <= root <= xb):
                print(f"    WARNING: root is outside the interval!")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")

# Let's also test with a valid bracket for comparison
print("\n" + "=" * 60)
print("Test with valid bracket for comparison")
print("=" * 60)
xa_valid = 0.5
xb_valid = 1.5
print(f"\nValid bracket: [{xa_valid}, {xb_valid}]")
print(f"f({xa_valid}) = {f(xa_valid)}")
print(f"f({xb_valid}) = {f(xb_valid)}")
print("Sign change present!\n")

print("Python API (bisect):")
try:
    root = scipy.optimize.bisect(f, xa_valid, xb_valid)
    print(f"  Result: {root}, f(root)={f(root)}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")

print("\nCython API (bisect):")
try:
    results = list(_zeros.loop_example('bisect', (a0,), (0.0, 0.0, 1.0), xa_valid, xb_valid, 0.01, 0.01, 50))
    print(f"  Results: {results}")
    if results:
        print(f"  f(root)={f(results[0])}")
except Exception as e:
    print(f"  Error: {type(e).__name__}: {e}")