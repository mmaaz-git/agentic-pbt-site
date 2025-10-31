from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

# Test scientific notation inputs that should work but fail
print("=" * 60)
print("Testing scientific notation CSS size parsing")
print("=" * 60)

# Test case 1: Scientific notation with 'pt' unit
result = resolver.size_to_pt("1e-5pt")
print(f"\nTest 1: Scientific notation 1e-5pt")
print(f"  Input:    '1e-5pt'")
print(f"  Output:   '{result}'")
print(f"  Expected: '0.00001pt' (or equivalent)")
print(f"  Actual value: {float(result.rstrip('pt'))}")

# Test case 2: Scientific notation with 'px' unit (should convert to pt)
result = resolver.size_to_pt("2.5e3px")
print(f"\nTest 2: Scientific notation 2.5e3px")
print(f"  Input:    '2.5e3px'")
print(f"  Output:   '{result}'")
print(f"  Expected: '1875pt' (2500 * 0.75 conversion)")
actual_val = float(result.rstrip('pt'))
print(f"  Actual value: {actual_val}")

# Test case 3: Compare with non-scientific notation equivalents
print("\n" + "=" * 60)
print("Comparison with equivalent non-scientific notation")
print("=" * 60)

# Test the same values without scientific notation
result_regular = resolver.size_to_pt("0.00001pt")
print(f"\nRegular notation equivalent of 1e-5pt:")
print(f"  Input:    '0.00001pt'")
print(f"  Output:   '{result_regular}'")
print(f"  Value:    {float(result_regular.rstrip('pt'))}")

result_regular = resolver.size_to_pt("2500px")
print(f"\nRegular notation equivalent of 2.5e3px:")
print(f"  Input:    '2500px'")
print(f"  Output:   '{result_regular}'")
print(f"  Value:    {float(result_regular.rstrip('pt'))}")

# Test case 4: Additional scientific notation formats
print("\n" + "=" * 60)
print("Additional scientific notation tests")
print("=" * 60)

test_cases = [
    "1.23e-2em",
    "5E+2pt",
    "3.14E-1px",
]

for test in test_cases:
    result = resolver.size_to_pt(test)
    print(f"\nInput:  '{test}'")
    print(f"Output: '{result}'")
    try:
        print(f"Value:  {float(result.rstrip('pt'))}")
    except:
        print(f"Value:  <unable to parse>")