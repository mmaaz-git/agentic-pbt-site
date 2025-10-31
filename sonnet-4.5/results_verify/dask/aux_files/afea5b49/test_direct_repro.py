import numpy as np
import numpy.strings as nps

s = '0'
arr = np.array([s])
result = nps.replace(arr, '0', '00')

print(f'Expected: "00"')
print(f'Got: "{result[0]}"')

# Check the dtype of arrays
print(f"\nOriginal array dtype: {arr.dtype}")
print(f"Result array dtype: {result.dtype}")

# Let's also test the fix example provided
print("\n--- Fix example comparison ---")
s = 'a'
arr = np.array([s])

result_ljust = nps.ljust(arr, 5)
print(f"ljust result dtype: {result_ljust.dtype}")

result_replace = nps.replace(arr, 'a', 'aaaaa')
print(f"replace result dtype: {result_replace.dtype}")
print(f"replace result: '{result_replace[0]}'")
print(f"Expected: 'aaaaa'")

# Verify the assertion fails
try:
    s = '0'
    arr = np.array([s])
    result = nps.replace(arr, '0', '00')
    assert result[0] == '00'
    print("\nAssertion passed (unexpected!)")
except AssertionError:
    print("\nAssertion failed as expected - bug confirmed!")