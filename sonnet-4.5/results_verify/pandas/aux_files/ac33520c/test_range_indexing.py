import numpy as np

# Test what happens when we use range objects as indexers
target = np.arange(10)

# Normal case
r1 = range(2, 5)
print(f"range(2, 5): {list(r1)}")
print(f"target[range(2, 5)]: {target[r1]}")
print(f"Length: {len(target[r1])}")
print()

# Empty range cases
r2 = range(5, 2)  # Empty with positive step
print(f"range(5, 2): {list(r2)}")
print(f"target[range(5, 2)]: {target[r2]}")
print(f"Length: {len(target[r2])}")
print()

r3 = range(1, 0, 1)  # Empty range
print(f"range(1, 0, 1): {list(r3)}")
print(f"target[range(1, 0, 1)]: {target[r3]}")
print(f"Length: {len(target[r3])}")
print()

# Compare with Python's len
print("Comparison with Python's len():")
for start, stop, step in [(2, 5, 1), (5, 2, 1), (1, 0, 1), (10, 0, 2)]:
    r = range(start, stop, step)
    print(f"range({start}, {stop}, {step}): len() = {len(r)}, len(target[r]) = {len(target[r])}")