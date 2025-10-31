import scipy.constants as const

# Basic test demonstrating the bug
all_keys = const.physical_constants.keys()
find_result = const.find(None)

print(f"physical_constants has {len(list(all_keys))} keys")
print(f"find(None) returns {len(find_result)} keys")
print(f"Missing: {len(list(all_keys)) - len(find_result)} keys")

# Test specific obsolete constant
obsolete_key = 'muon Compton wavelength over 2 pi'
print(f"\nObsolete constant '{obsolete_key}':")
print(f"  Exists in physical_constants: {obsolete_key in const.physical_constants}")
print(f"  Found by find(None): {obsolete_key in find_result}")

# Test search for 'muon'
muon_in_physical_constants = [k for k in const.physical_constants.keys() if 'muon' in k.lower()]
muon_from_find = const.find('muon')
print(f"\nSearching for 'muon':")
print(f"  Keys in physical_constants containing 'muon': {len(muon_in_physical_constants)}")
print(f"  Keys returned by find('muon'): {len(muon_from_find)}")
print(f"  Missing from find: {len(muon_in_physical_constants) - len(muon_from_find)} keys")

# This should succeed according to documentation but fails
assert len(list(all_keys)) == len(find_result), f"find(None) should return all keys but returns {len(find_result)} instead of {len(list(all_keys))}"