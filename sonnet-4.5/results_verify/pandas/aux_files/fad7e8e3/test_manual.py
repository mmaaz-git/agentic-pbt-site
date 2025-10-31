from pandas.io.formats.format import format_percentiles

# Test the case from the bug report
percentiles = [0.01, 0.0100001]
result = format_percentiles(percentiles)

print(f"Input:  {percentiles}")
print(f"Output: {result}")
print(f"Unique inputs:  {len(set(percentiles))}")
print(f"Unique outputs: {len(set(result))}")

# Also test the case found by hypothesis
percentiles2 = [0.0, 2.2250738585072014e-308]
result2 = format_percentiles(percentiles2)

print(f"\nSecond test:")
print(f"Input:  {percentiles2}")
print(f"Output: {result2}")
print(f"Unique inputs:  {len(set(percentiles2))}")
print(f"Unique outputs: {len(set(result2))}")