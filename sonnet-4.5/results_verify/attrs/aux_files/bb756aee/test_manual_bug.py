from dask.utils import format_bytes

pib = 2**50

test_cases = [
    999 * pib,
    1000 * pib,
    1023 * pib,
]

print("Testing format_bytes output length:")
print("-" * 60)

for val in test_cases:
    result = format_bytes(val)
    print(f"{val:20d} -> '{result:12s}' (len={len(result)})")

print("\n2^60 = ", 2**60)
print("All test values are < 2^60:", all(val < 2**60 for val in test_cases))