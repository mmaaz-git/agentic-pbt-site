import pandas.tseries.frequencies as freq

# Test reflexivity for all frequency types
test_freqs = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]

print("Testing reflexivity property: is_subperiod(X, X) and is_superperiod(X, X) should always return True")
print("=" * 80)

for f in test_freqs:
    sub_result = freq.is_subperiod(f, f)
    super_result = freq.is_superperiod(f, f)

    if not sub_result or not super_result:
        print(f"FAIL - Frequency: '{f}'")
        print(f"  is_subperiod('{f}', '{f}') = {sub_result} (Expected: True)")
        print(f"  is_superperiod('{f}', '{f}') = {super_result} (Expected: True)")
    else:
        print(f"OK   - Frequency: '{f}' - Both functions correctly return True")
    print()

print("=" * 80)
print("SUMMARY:")
print("Frequencies where reflexivity is violated (bugs):")
failing_sub = [f for f in test_freqs if not freq.is_subperiod(f, f)]
failing_super = [f for f in test_freqs if not freq.is_superperiod(f, f)]
print(f"  is_subperiod reflexivity failures: {', '.join(failing_sub) if failing_sub else 'None'}")
print(f"  is_superperiod reflexivity failures: {', '.join(failing_super) if failing_super else 'None'}")