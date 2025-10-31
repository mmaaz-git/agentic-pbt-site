from pandas.io.formats.format import format_percentiles

percentile = 0.8899967487632947
formatted = format_percentiles([percentile])

print(f"Input: {percentile}")
print(f"Percent value: {percentile * 100}%")
print(f"Output: {formatted[0]}")

is_integer = abs((percentile * 100) - round(percentile * 100)) < 1e-10
print(f"Is integer percent: {is_integer}")

# Check if decimal is present
has_decimal = '.' in formatted[0]
print(f"Has decimal in output: {has_decimal}")

if not has_decimal:
    print("ERROR: Non-integer should have decimal place")
else:
    print("OK: Output has decimal place")