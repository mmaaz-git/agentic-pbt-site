import scipy.constants as sc

# Check if 'Planck constant over 2 pi' is from an older version
test_key = 'Planck constant over 2 pi'

# Check which version it's from
print(f"Checking which version '{test_key}' is from:")

# Get all keys from each version
current_keys = set(sc._codata._current_constants.keys())
all_keys = set(sc.physical_constants.keys())

# Find keys that are only in physical_constants but not in current
older_keys = all_keys - current_keys

print(f"\nTotal keys in physical_constants: {len(all_keys)}")
print(f"Keys in current (2022) constants: {len(current_keys)}")
print(f"Keys only in older versions: {len(older_keys)}")

print(f"\n'{test_key}' is in older versions only: {test_key in older_keys}")

# Show some examples of older keys
print("\nSome examples of keys only in older versions:")
for i, key in enumerate(sorted(older_keys)[:10]):
    print(f"  - {key}")