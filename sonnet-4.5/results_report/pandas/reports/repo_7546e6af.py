from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test with 'Y-JAN' frequency
freq = 'Y-JAN'
print(f"Testing frequency: '{freq}'")
print(f"is_superperiod('{freq}', '{freq}') = {is_superperiod(freq, freq)}")
print(f"is_subperiod('{freq}', '{freq}') = {is_subperiod(freq, freq)}")

# Check if symmetry property holds
super_result = is_superperiod(freq, freq)
sub_result = is_subperiod(freq, freq)

if super_result == sub_result:
    print(f"\n✓ Symmetry property holds: both return {super_result}")
else:
    print(f"\n✗ Symmetry violated!")
    print(f"  Expected: is_superperiod(a, b) == is_subperiod(b, a)")
    print(f"  Got: is_superperiod('{freq}', '{freq}') = {super_result}")
    print(f"       is_subperiod('{freq}', '{freq}') = {sub_result}")

# Test other annual frequencies
print("\n" + "="*50)
print("Testing other annual frequencies:")
annual_freqs = ['Y', 'Y-JAN', 'Y-FEB', 'Y-MAR', 'Y-DEC']

for freq in annual_freqs:
    super_res = is_superperiod(freq, freq)
    sub_res = is_subperiod(freq, freq)
    match = "✓" if super_res == sub_res else "✗"
    print(f"{match} '{freq}': is_superperiod={super_res}, is_subperiod={sub_res}")

# Demonstrate the expected behavior (symmetry property)
print("\n" + "="*50)
print("Demonstrating symmetry violation between different pairs:")
test_pairs = [('Y-JAN', 'Y-FEB'), ('Y', 'Y-JAN'), ('Y-MAR', 'Y')]

for source, target in test_pairs:
    super_res = is_superperiod(source, target)
    sub_res = is_subperiod(target, source)
    match = "✓" if super_res == sub_res else "✗"
    print(f"{match} is_superperiod('{source}', '{target}') = {super_res}")
    print(f"    is_subperiod('{target}', '{source}') = {sub_res}")
    if super_res != sub_res:
        print(f"    ^ Symmetry violated!")
    print()