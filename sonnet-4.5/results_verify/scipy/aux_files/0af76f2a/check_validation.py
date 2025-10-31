import scipy.signal.windows as w
import numpy as np

# Test how other window functions handle invalid parameters

print("Testing parameter validation in other scipy.signal.windows functions:")
print("="*60)

# Test 1: kaiser window with negative beta
print("\n1. Kaiser window with negative beta (should be non-negative):")
try:
    result = w.kaiser(10, beta=-5)
    print(f"   Result: {result}")
    print(f"   Contains NaN: {np.any(np.isnan(result))}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")

# Test 2: tukey window with alpha > 1 (should be [0, 1])
print("\n2. Tukey window with alpha > 1 (should be in [0,1]):")
try:
    result = w.tukey(10, alpha=1.5)
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")

# Test 3: tukey window with negative alpha
print("\n3. Tukey window with negative alpha (should be in [0,1]):")
try:
    result = w.tukey(10, alpha=-0.5)
    print(f"   Result: {result}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")

# Test 4: gaussian window with negative std (should be positive)
print("\n4. Gaussian window with negative std (should be positive):")
try:
    result = w.gaussian(10, std=-1)
    print(f"   Result: {result}")
    print(f"   Contains NaN: {np.any(np.isnan(result))}")
except Exception as e:
    print(f"   Exception raised: {type(e).__name__}: {e}")

# Test 5: Check if M validation is consistent
print("\n5. Testing M parameter validation (negative M):")
functions = [
    ('taylor', lambda: w.taylor(-5)),
    ('kaiser', lambda: w.kaiser(-5, beta=5)),
    ('tukey', lambda: w.tukey(-5)),
    ('gaussian', lambda: w.gaussian(-5, std=1)),
    ('hamming', lambda: w.hamming(-5))
]

for name, func in functions:
    try:
        result = func()
        print(f"   {name}: Returned {result}")
    except Exception as e:
        print(f"   {name}: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Observations:")
print("- Most functions validate M and raise ValueError for negative values")
print("- Parameter validation for other arguments is inconsistent")
print("- Some functions silently produce unexpected results for invalid params")