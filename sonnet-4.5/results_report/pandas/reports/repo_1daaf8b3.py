from pandas.io.formats.format import format_percentiles

percentile = 1.401298464324817e-45
result = format_percentiles([percentile])

print(f"Input percentile: {percentile}")
print(f"Output: {result[0]}")
print(f"Is percentile zero: {percentile == 0.0}")
print(f"Is percentile * 100 an integer: {(percentile * 100).is_integer()}")
print()
print("According to docstring:")
print("1. 'no entry is *rounded* to 0% or 100%' (unless already equal to it)")
print("2. 'Any non-integer is always rounded to at least 1 decimal place'")
print()
print(f"Bug: Non-zero value {percentile} is formatted as '{result[0]}' with no decimal")