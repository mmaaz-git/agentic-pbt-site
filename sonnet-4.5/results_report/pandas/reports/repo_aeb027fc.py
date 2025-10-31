from pandas.io.excel._util import _excel2num

# Test with empty string
print("Testing _excel2num with empty string:")
result1 = _excel2num('')
print(f"_excel2num('') = {result1}")

# Test with whitespace-only string
print("\nTesting _excel2num with whitespace-only string:")
result2 = _excel2num('   ')
print(f"_excel2num('   ') = {result2}")

# Test with valid column names for comparison
print("\nTesting _excel2num with valid column names:")
result3 = _excel2num('A')
print(f"_excel2num('A') = {result3}")

result4 = _excel2num('B')
print(f"_excel2num('B') = {result4}")

result5 = _excel2num('Z')
print(f"_excel2num('Z') = {result5}")

result6 = _excel2num('AA')
print(f"_excel2num('AA') = {result6}")

# Test with tab character
print("\nTesting _excel2num with tab character:")
result7 = _excel2num('\t')
print(f"_excel2num('\\t') = {result7}")

# Test how this affects _range2cols
from pandas.io.excel._util import _range2cols

print("\nTesting how empty strings affect _range2cols:")
result8 = _range2cols(',')  # This should have an empty string between commas
print(f"_range2cols(',') = {result8}")

result9 = _range2cols('A,,B')  # Empty column between A and B
print(f"_range2cols('A,,B') = {result9}")