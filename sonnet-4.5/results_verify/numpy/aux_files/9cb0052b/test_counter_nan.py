from collections import Counter
import math

# Test how Counter handles NaN values
lst = [float('nan'), float('nan'), 1.0, 1.0]
counter = Counter(lst)

print("Input list:", lst)
print("Counter result:", dict(counter))
print()

# Check individual counts
for key, count in counter.items():
    if isinstance(key, float) and math.isnan(key):
        print(f"NaN: count = {count}")
    else:
        print(f"{key}: count = {count}")

print()
print("NaN equality test:")
print(f"float('nan') == float('nan'): {float('nan') == float('nan')}")
print(f"math.nan == math.nan: {math.nan == math.nan}")

# Test with same NaN object
nan_obj = float('nan')
lst2 = [nan_obj, nan_obj, 1.0, 1.0]
counter2 = Counter(lst2)
print("\nUsing same NaN object:")
print("Input list:", lst2)
print("Counter result:", dict(counter2))
for key, count in counter2.items():
    if isinstance(key, float) and math.isnan(key):
        print(f"NaN: count = {count}")
    else:
        print(f"{key}: count = {count}")