import pandas as pd
import numpy as np
import traceback

print("Testing pd.cut() with underflow values...")
print("-" * 50)

# Test case from the bug report
x = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.2250738585072014e-308, -1.0]
print(f"Input array: {x}")
print(f"Bins: 2")
print(f"Special value in array: {2.2250738585072014e-308:.5e}")
print()

try:
    result = pd.cut(x, bins=2)
    print("Result:", result)
    print("Function succeeded unexpectedly!")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Let's also check what happens in the _round_frac function
print("\n" + "-" * 50)
print("Testing np.around with large digits value...")
test_val = 2.2250738585072014e-308
# Simulate what _round_frac would do
frac, whole = np.modf(test_val)
digits = -int(np.floor(np.log10(abs(frac)))) - 1 + 3  # precision=3 is default
print(f"For value {test_val:.5e}:")
print(f"  frac={frac}, whole={whole}")
print(f"  Computed digits: {digits}")
rounded = np.around(test_val, digits)
print(f"  np.around({test_val:.5e}, {digits}) = {rounded}")
print(f"  Is result NaN? {np.isnan(rounded)}")