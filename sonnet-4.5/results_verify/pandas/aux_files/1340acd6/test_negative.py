import pandas as pd
import traceback

print("Test 2: Negative subnormal values")
print("-" * 40)

values = [0.0, -2.225e-313]
s = pd.Series(values)

print(f"Input values: {values}")

try:
    result = pd.cut(s, bins=2)
    print(f"Result: {result.tolist()}")
    print(f"All values are NaN: {result.isna().all()}")
except ValueError as e:
    print(f"ValueError raised: {e}")
    print("✓ Confirmed: ValueError raised for negative subnormal values")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")
    traceback.print_exc()