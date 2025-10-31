from pandas.io.formats.format import format_percentiles

# Test case from bug report
percentiles = [0.01, 0.0100001]
result = format_percentiles(percentiles)

print(f"Input:  {percentiles}")
print(f"Output: {result}")
print(f"Unique inputs:  {len(set(percentiles))}")
print(f"Unique outputs: {len(set(result))}")
print()
print("Expected: 2 unique outputs (since we have 2 unique inputs)")
print("Actual: Only 1 unique output - both values formatted as '1%'")
print()
print("This violates the documented guarantee that:")
print("'if any two elements of percentiles differ, they remain different after rounding'")