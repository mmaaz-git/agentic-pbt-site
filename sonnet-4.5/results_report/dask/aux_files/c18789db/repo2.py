from dask.sizeof import sizeof
import random

# Test with a list of 100 items (much larger than 10)
lst = list(range(100))

print("Testing sizeof on a list with 100 items:")
results = []
for i in range(20):
    result = sizeof(lst)
    results.append(result)
    if i < 10:
        print(f"Call {i+1}: {result}")

print(f"...")
print(f"\nAll unique values after 20 calls: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")

# Test with a dict of 100 items
d = {i: i*2 for i in range(100)}

print("\n\nTesting sizeof on a dict with 100 items:")
results = []
for i in range(20):
    result = sizeof(d)
    results.append(result)
    if i < 10:
        print(f"Call {i+1}: {result}")

print(f"...")
print(f"\nAll unique values after 20 calls: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")

# Test with strings to see if the issue is with content
lst_str = [f"string_{i}" for i in range(50)]

print("\n\nTesting sizeof on a list of 50 strings:")
results = []
for i in range(20):
    result = sizeof(lst_str)
    results.append(result)
    if i < 10:
        print(f"Call {i+1}: {result}")

print(f"...")
print(f"\nAll unique values after 20 calls: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")