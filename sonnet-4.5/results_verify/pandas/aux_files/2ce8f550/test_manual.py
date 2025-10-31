import pandas.core.algorithms as algorithms

values = ['', '\x00']
codes, uniques = algorithms.factorize(values)

print(f"Input:         {values}")
print(f"Input (repr):  {[repr(v) for v in values]}")
print(f"Codes:         {codes}")
print(f"Uniques:       {list(uniques)}")
print(f"Uniques (repr): {[repr(u) for u in uniques]}")
reconstructed = uniques.take(codes)
print(f"Reconstructed: {list(reconstructed)}")
print(f"Reconstructed (repr): {[repr(r) for r in reconstructed]}")
print()

print("Checking assertions:")
try:
    assert values[1] == '\x00', "Original value is null character"
    print("✓ Original value is null character")
except AssertionError as e:
    print(f"✗ {e}")

try:
    assert uniques.take(codes)[1] == '\x00', "Round-trip should preserve null character"
    print("✓ Round-trip preserves null character")
except AssertionError as e:
    print(f"✗ {e}")

print("\nChecking if codes are same for both values:")
print(f"Code for '':    {codes[0]}")
print(f"Code for '\\x00': {codes[1]}")
print(f"Codes are same? {codes[0] == codes[1]}")