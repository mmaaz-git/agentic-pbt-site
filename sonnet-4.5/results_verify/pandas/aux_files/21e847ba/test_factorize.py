import numpy as np
from pandas import factorize, Index, Categorical
from pandas.core.dtypes.dtypes import CategoricalDtype

# Test what factorize does with '' and '\x00'
data = ['', '\x00']
arr = np.array(data, dtype=object)

# This is what happens in the categorize path
codes, categories = factorize(arr, sort=False)
print(f"Original data: {data}")
print(f"Codes after factorize: {codes}")
print(f"Categories after factorize: {list(categories)}")
print(f"Are '' and '\\x00' treated as same category? {codes[0] == codes[1]}")

# Now let's see what happens with the categorical
dtype = CategoricalDtype(categories=Index(categories), ordered=False)
cat = Categorical._simple_new(codes, dtype)
print(f"\nCategorical codes: {cat.codes}")
print(f"Categorical categories: {list(cat.categories)}")

# Let's also test hash_object_array directly
from pandas._libs.hashing import hash_object_array

hash_key = "0123456789123456"
encoding = "utf8"

direct_hash = hash_object_array(arr, hash_key, encoding)
print(f"\nDirect hash_object_array: {direct_hash}")
print(f"Are direct hashes equal? {direct_hash[0] == direct_hash[1]}")