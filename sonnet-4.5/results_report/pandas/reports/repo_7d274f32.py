import pandas.io._util as util

# Get the arrow dtype mapping
pa = util.import_optional_dependency("pyarrow")
mapping = util._arrow_dtype_mapping()

# Count the actual number of keys
actual_keys = len(mapping)

# Count expected keys from source code (lines 32-45 = 14 entries)
expected_keys = 14

print(f"Number of keys in mapping: {actual_keys}")
print(f"Expected keys (from source): {expected_keys}")
print()

# Demonstrate the duplicate key issue
print("Issue: pa.string() appears twice in the dictionary literal:")
print("  - Line 41: pa.string(): pd.StringDtype()")
print("  - Line 44: pa.string(): pd.StringDtype() (duplicate)")
print()
print("Due to Python's dictionary semantics, the second occurrence")
print("overwrites the first, resulting in only 13 unique keys.")
print()

# Show all keys in the mapping
print("Keys in the mapping:")
for i, key in enumerate(mapping.keys(), 1):
    print(f"  {i}. {key}")
print()

# Verify pa.string() is only there once despite being defined twice
string_type_count = sum(1 for k in mapping.keys() if str(k) == "string")
print(f"Number of 'string' type keys in final mapping: {string_type_count}")