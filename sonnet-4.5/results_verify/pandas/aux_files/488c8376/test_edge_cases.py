import pandas as pd
import numpy as np

# More edge cases to understand the pattern
test_cases = [
    (['0', '0\x00'], "No duplicate - works correctly"),
    (['0', '0', '0\x00'], "Duplicate '0' then '0\\x00' - FAILS"),
    (['0\x00', '0', '0'], "'0\\x00' first then duplicates - works"),
    (['0', '0\x00', '0'], "'0', '0\\x00', '0' - interesting case"),
    (['0', '1', '0', '0\x00'], "Other values between duplicates"),
    (['a', 'a', 'a\x00'], "Same pattern with 'a'"),
    (['', '', '\x00'], "Empty string pattern"),
]

print("Testing edge cases for Index.unique() bug:")
print("=" * 70)

for test, description in test_cases:
    idx = pd.Index(test)
    unique_result = idx.unique()
    expected = sorted(set(test))

    print(f"\n{description}")
    print(f"Input:    {[repr(x) for x in test]}")
    print(f"Expected: {[repr(x) for x in expected]}")
    print(f"Actual:   {[repr(x) for x in unique_result]}")
    print(f"BUG:      {'YES - DATA LOSS!' if len(expected) != len(unique_result) else 'No'}")