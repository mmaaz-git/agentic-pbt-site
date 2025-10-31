from dask.sizeof import sizeof

# Create a list with items of different sizes to make the non-determinism obvious
# Using strings of different lengths
lst = ["a" * i for i in range(1, 21)]  # 20 strings of lengths 1 to 20

print("Testing sizeof on a list with 20 strings of varying lengths:")
print(f"List contents: ['a', 'aa', 'aaa', ..., 'a'*20]")
print()

results = []
for i in range(10):
    result = sizeof(lst)
    results.append(result)
    print(f"Call {i+1}: {result}")

print(f"\nUnique values: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")

# Also test with a dict
d = {i: "x" * i for i in range(1, 21)}  # Dict with values of different sizes

print("\n\nTesting sizeof on a dict with 20 entries having values of varying lengths:")
print(f"Dict contents: {{1: 'x', 2: 'xx', 3: 'xxx', ..., 20: 'x'*20}}")
print()

results = []
for i in range(10):
    result = sizeof(d)
    results.append(result)
    print(f"Call {i+1}: {result}")

print(f"\nUnique values: {sorted(set(results))}")
print(f"Number of unique values: {len(set(results))}")
print(f"Deterministic? {len(set(results)) == 1}")