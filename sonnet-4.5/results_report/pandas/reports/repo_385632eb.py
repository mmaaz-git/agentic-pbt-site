from pandas.core.methods.describe import format_percentiles

# Test case 1: Two different values collapse to same format
percentiles = [0.0, 3.6340605919844266e-284]
result = format_percentiles(percentiles)

print("Test 1: Uniqueness violation")
print(f"Input percentiles: {percentiles}")
print(f"Unique input count: {len(set(percentiles))}")
print(f"Output: {result}")
print(f"Unique output count: {len(set(result))}")

# Verify the issue
try:
    assert len(set(percentiles)) == 2
    assert len(set(result)) == 2
except AssertionError as e:
    print(f"AssertionError: Expected 2 unique outputs, but got {len(set(result))}: {set(result)}")

print("\n" + "="*60 + "\n")

# Test case 2: Non-zero value rounds to 0%
percentiles2 = [1.401298464324817e-45]
result2 = format_percentiles(percentiles2)

print("Test 2: Non-zero rounding to 0%")
print(f"Input: {percentiles2}")
print(f"Output: {result2}")

# Verify the issue
try:
    assert result2[0] != '0%', f"Non-zero percentile {percentiles2[0]} rounded to 0%"
except AssertionError as e:
    print(f"AssertionError: {e}")