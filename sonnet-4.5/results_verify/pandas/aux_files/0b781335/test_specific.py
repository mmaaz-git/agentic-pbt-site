import pandas as pd

# Test the specific failing example from the bug report
values = [0.0, 1.1125369292536007e-308]
x = pd.Series(values)

print(f"Testing specific example: {values}")
result = pd.cut(x, bins=2)

if result.notna().sum() == x.notna().sum():
    print("TEST PASSED: All values were binned")
else:
    print(f"TEST FAILED: Data loss - {x.notna().sum()} valid inputs became {result.notna().sum()} valid outputs")
    print(f"Result: {result.tolist()}")