import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd

# Test case 1: Empty string vs null character
print("Test 1: Empty string vs null character")
arr = ['', '\x00']
cat = pd.Categorical(arr)

print(f"Input:      {repr(arr)}")
print(f"Categories: {repr(list(cat.categories))}")
print(f"Codes:      {cat.codes.tolist()}")

reconstructed = list(cat.categories[cat.codes])
print(f"Output:     {repr(reconstructed)}")
print(f"Match:      {reconstructed == arr}")
print()

# Test case 2: String with and without trailing null
print("Test 2: String with and without trailing null")
arr2 = ['a', 'a\x00']
cat2 = pd.Categorical(arr2)

print(f"Input:      {repr(arr2)}")
print(f"Categories: {repr(list(cat2.categories))}")
print(f"Codes:      {cat2.codes.tolist()}")

reconstructed2 = list(cat2.categories[cat2.codes])
print(f"Output:     {repr(reconstructed2)}")
print(f"Match:      {reconstructed2 == arr2}")
print()

# Test case 3: Null character vs other control character (this works correctly)
print("Test 3: Null character vs other control character")
arr3 = ['\x00', '\x01']
cat3 = pd.Categorical(arr3)

print(f"Input:      {repr(arr3)}")
print(f"Categories: {repr(list(cat3.categories))}")
print(f"Codes:      {cat3.codes.tolist()}")

reconstructed3 = list(cat3.categories[cat3.codes])
print(f"Output:     {repr(reconstructed3)}")
print(f"Match:      {reconstructed3 == arr3}")