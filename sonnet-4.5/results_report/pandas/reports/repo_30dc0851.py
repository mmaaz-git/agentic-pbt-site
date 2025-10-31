import pandas as pd

values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.15625]
result = pd.cut(values, bins=2, precision=3)

val = 1.15625
cat = result[9]

print(f"Value: {val}")
print(f"Assigned interval: {cat}")
print(f"Value in interval? {val in cat}")
print()
print("Details:")
print(f"  Interval left boundary: {cat.left}")
print(f"  Interval right boundary: {cat.right}")
print(f"  Is value > right boundary? {val > cat.right}")