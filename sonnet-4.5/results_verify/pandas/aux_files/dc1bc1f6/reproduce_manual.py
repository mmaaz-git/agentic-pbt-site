import pandas.core.arrays as arrays

values = ['', '\x00']
cat = arrays.Categorical(values)

print(f"Input: {repr(values)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes: {list(cat.codes)}")

reconstructed = [cat.categories[c] if c != -1 else None for c in cat.codes]
print(f"Original:      {repr(values)}")
print(f"Reconstructed: {repr(reconstructed)}")
print(f"Data preserved: {values == reconstructed}")

# Additional tests to understand the behavior
print("\n--- Additional analysis ---")
print(f"Empty string == null char: {'' == '\x00'}")
print(f"Empty string length: {len('')}")
print(f"Null char length: {len('\x00')}")
print(f"Number of unique categories: {len(cat.categories)}")
print(f"Expected unique categories: 2")

# Test accessing individual elements
print(f"\ncat[0] = {repr(cat[0])}")
print(f"cat[1] = {repr(cat[1])}")
print(f"values[0] = {repr(values[0])}")
print(f"values[1] = {repr(values[1])}")