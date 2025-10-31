import scipy.constants as sc

# First test the basic reproduction
key = 'Planck constant over 2 pi'

print(f"sc.physical_constants['{key}'] = {sc.physical_constants[key]}")

results = sc.find('Planck constant over 2 pi')
print(f"sc.find('Planck constant over 2 pi') = {results}")

print(f"Results length: {len(results)}")
if len(results) == 0:
    print("BUG CONFIRMED: find() cannot locate this constant!")

print(f"\n_current_constants has {len(sc._codata._current_constants)} keys")
print(f"physical_constants has {len(sc.physical_constants)} keys")
print(f"{len(sc.physical_constants) - len(sc._codata._current_constants)} constants are inaccessible via find()")