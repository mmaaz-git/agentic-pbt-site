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