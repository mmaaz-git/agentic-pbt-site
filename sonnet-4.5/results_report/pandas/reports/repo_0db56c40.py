import pandas as pd

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.84375]
x = pd.Series(values)

result, bins = pd.cut(x, bins=2, retbins=True, precision=3)

print(f"Bins returned via retbins: {bins}")
print(f"Interval categories: {result.cat.categories}")

categories = result.cat.categories
for i, interval in enumerate(categories):
    print(f"\nInterval {i}:")
    print(f"  String representation: {interval}")
    print(f"  Actual left boundary: {interval.left}")
    print(f"  Actual right boundary: {interval.right}")
    print(f"  Expected from bins: [{bins[i]}, {bins[i+1]}]")

    if interval.left != bins[i]:
        print(f"  MISMATCH: {interval.left} != {bins[i]}")
    if interval.right != bins[i+1]:
        print(f"  MISMATCH: {interval.right} != {bins[i+1]}")