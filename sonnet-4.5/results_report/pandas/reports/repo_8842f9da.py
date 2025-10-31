from pandas.io.excel._util import _excel2num, _range2cols

# Test empty string
print("Testing _excel2num(''):")
try:
    result = _excel2num("")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test whitespace-only string
print("\nTesting _excel2num('   '):")
try:
    result = _excel2num("   ")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test tab character
print("\nTesting _excel2num('\\t'):")
try:
    result = _excel2num("\t")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test newline character
print("\nTesting _excel2num('\\n'):")
try:
    result = _excel2num("\n")
    print(f"  Returned: {result}")
except ValueError as e:
    print(f"  Raised ValueError: {e}")

# Test downstream impact in _range2cols
print("\n\nDownstream impact in _range2cols:")
print(f"_range2cols('A,,C'): {_range2cols('A,,C')}")
print(f"_range2cols(',A'): {_range2cols(',A')}")
print(f"_range2cols('A,'): {_range2cols('A,')}")
print(f"_range2cols('   ,A'): {_range2cols('   ,A')}")

# Test with valid inputs for comparison
print("\n\nValid inputs for comparison:")
print(f"_excel2num('A'): {_excel2num('A')}")
print(f"_excel2num('B'): {_excel2num('B')}")
print(f"_excel2num('Z'): {_excel2num('Z')}")
print(f"_excel2num('AA'): {_excel2num('AA')}")
print(f"_excel2num('AB'): {_excel2num('AB')}")