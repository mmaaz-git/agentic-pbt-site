import scipy.constants as sc

# Check if 'Planck constant over 2 pi' exists in both
key1 = 'Planck constant over 2 pi'
key2 = 'reduced Planck constant'

print(f"'{key1}' in physical_constants: {key1 in sc.physical_constants}")
print(f"'{key1}' in _current_constants: {key1 in sc._codata._current_constants}")

print(f"\n'{key2}' in physical_constants: {key2 in sc.physical_constants}")
print(f"'{key2}' in _current_constants: {key2 in sc._codata._current_constants}")

# Check what these are
if key1 in sc.physical_constants:
    print(f"\n{key1} = {sc.physical_constants[key1]}")
if key2 in sc.physical_constants:
    print(f"{key2} = {sc.physical_constants[key2]}")

# Are they the same value?
if key1 in sc.physical_constants and key2 in sc.physical_constants:
    val1 = sc.physical_constants[key1][0]
    val2 = sc.physical_constants[key2][0]
    print(f"\nSame value? {val1 == val2}")