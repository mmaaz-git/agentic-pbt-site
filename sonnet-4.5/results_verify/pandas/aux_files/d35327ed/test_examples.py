from pandas.io.formats.format import format_percentiles

# Test examples from the documentation
print("Testing documentation examples:")
print()

# Example 1
result1 = format_percentiles([0.01999, 0.02001, 0.5, 0.666666, 0.9999])
expected1 = ['1.999%', '2.001%', '50%', '66.667%', '99.99%']
print(f"Example 1:")
print(f"  Input:    [0.01999, 0.02001, 0.5, 0.666666, 0.9999]")
print(f"  Expected: {expected1}")
print(f"  Actual:   {result1}")
print(f"  Match:    {result1 == expected1}")
print()

# Example 2
result2 = format_percentiles([0, 0.5, 0.02001, 0.5, 0.666666, 0.9999])
expected2 = ['0%', '50%', '2.0%', '50%', '66.67%', '99.99%']
print(f"Example 2:")
print(f"  Input:    [0, 0.5, 0.02001, 0.5, 0.666666, 0.9999]")
print(f"  Expected: {expected2}")
print(f"  Actual:   {result2}")
print(f"  Match:    {result2 == expected2}")
print()

# Test the note about non-integers having at least 1 decimal place
print("Testing claim: 'Any non-integer is always rounded to at least 1 decimal place'")
print()

# Test case from bug report
test_val = 0.8899967487632947
result = format_percentiles([test_val])
print(f"Test value: {test_val} ({test_val * 100}%)")
print(f"Result:     {result[0]}")
print(f"Has decimal: {'.' in result[0]}")
print()

# Let's check what "non-integer" means - is 0.5 (50%) considered integer?
print("Is 0.5 (50%) considered integer?")
result_50 = format_percentiles([0.5])
print(f"  format_percentiles([0.5]) = {result_50}")
print(f"  Has decimal: {'.' in result_50[0]}")
print()

# What about 0.25 (25%)?
print("Is 0.25 (25%) considered integer?")
result_25 = format_percentiles([0.25])
print(f"  format_percentiles([0.25]) = {result_25}")
print(f"  Has decimal: {'.' in result_25[0]}")