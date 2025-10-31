import pandas.tseries.frequencies as freq

test_freqs = ["M", "Q", "Y"]

for f in test_freqs:
    sub_result = freq.is_subperiod(f, f)
    super_result = freq.is_superperiod(f, f)
    print(f"is_subperiod('{f}', '{f}') = {sub_result}")
    print(f"is_superperiod('{f}', '{f}') = {super_result}")
    print()

# Test all valid frequencies
all_freqs = ["D", "h", "B", "C", "M", "W", "Q", "Y", "min", "s", "ms", "us", "ns"]
print("Testing all frequencies for reflexivity violations:")
print("-" * 50)

subperiod_failures = []
superperiod_failures = []

for f in all_freqs:
    sub = freq.is_subperiod(f, f)
    super = freq.is_superperiod(f, f)

    if not sub:
        subperiod_failures.append(f)
    if not super:
        superperiod_failures.append(f)

    status_sub = "✓" if sub else "✗"
    status_super = "✓" if super else "✗"
    print(f"{f:5} - is_subperiod: {status_sub}  is_superperiod: {status_super}")

print("\n" + "=" * 50)
print("SUMMARY:")
if subperiod_failures:
    print(f"is_subperiod reflexivity failures: {', '.join(subperiod_failures)}")
else:
    print("is_subperiod: All frequencies pass reflexivity test")

if superperiod_failures:
    print(f"is_superperiod reflexivity failures: {', '.join(superperiod_failures)}")
else:
    print("is_superperiod: All frequencies pass reflexivity test")