from pandas.io.excel._util import _excel2num, _range2cols

# Test _excel2num with empty string
print("Testing _excel2num with empty string:")
try:
    result = _excel2num('')
    print(f"_excel2num('') = {result}")
except ValueError as e:
    print(f"_excel2num('') raised ValueError: {e}")

# Test _excel2num with whitespace-only string
print("\nTesting _excel2num with whitespace-only string:")
try:
    result = _excel2num('   ')
    print(f"_excel2num('   ') = {result}")
except ValueError as e:
    print(f"_excel2num('   ') raised ValueError: {e}")

# Test _range2cols with trailing comma
print("\nTesting _range2cols with trailing comma:")
result = _range2cols('A,')
print(f"_range2cols('A,') = {result}")

# Test _range2cols with only comma
print("\nTesting _range2cols with only comma:")
result = _range2cols(',')
print(f"_range2cols(',') = {result}")

# Test _range2cols with multiple trailing commas
print("\nTesting _range2cols with multiple trailing commas:")
result = _range2cols('A,,')
print(f"_range2cols('A,,') = {result}")

# Test _range2cols with comma between valid columns
print("\nTesting _range2cols with empty element between valid columns:")
result = _range2cols('A,,B')
print(f"_range2cols('A,,B') = {result}")

# Test that valid inputs still work
print("\nTesting valid inputs:")
print(f"_excel2num('A') = {_excel2num('A')}")
print(f"_excel2num('Z') = {_excel2num('Z')}")
print(f"_excel2num('AA') = {_excel2num('AA')}")
print(f"_range2cols('A,B,C') = {_range2cols('A,B,C')}")
print(f"_range2cols('A:C') = {_range2cols('A:C')}")