import pandas as pd
import numpy as np

print("Testing the reported bug...")
print("=" * 60)

# Test case from bug report
data = [1.1125369292536007e-308, -1.0]
print(f"Input data: {data}")
print(f"Number of bins: 2")

try:
    result = pd.cut(data, bins=2)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Testing what happens with _round_frac on extreme values...")

# Test the _round_frac function behavior
def test_round_frac(x, precision=3):
    """Mimics the _round_frac function"""
    if not np.isfinite(x) or x == 0:
        return x
    else:
        frac, whole = np.modf(x)
        if whole == 0:
            digits = -int(np.floor(np.log10(abs(frac)))) - 1 + precision
            print(f"  For x={x}, frac={frac}, calculated digits={digits}")
        else:
            digits = precision
            print(f"  For x={x}, whole={whole}, using digits={digits}")
        result = np.around(x, digits)
        print(f"  np.around({x}, {digits}) = {result}")
        return result

# Test with the extreme value
extreme_val = 1.1125369292536007e-308
print(f"\nTesting _round_frac logic with extreme value: {extreme_val}")
test_round_frac(extreme_val, precision=3)

# Test what np.around returns with large digits
print("\nTesting np.around with large digit counts:")
test_vals = [(extreme_val, 310), (extreme_val, 100), (extreme_val, 50), (extreme_val, 15)]
for val, digits in test_vals:
    result = np.around(val, digits)
    print(f"  np.around({val}, {digits}) = {result}, isnan={np.isnan(result)}")