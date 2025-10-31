from pandas.io.formats.css import CSSResolver

# Create a resolver instance
resolver = CSSResolver()

# Test cases with scientific notation
print("Testing scientific notation CSS size parsing:")
print("=" * 50)

# Test case 1: Small value in scientific notation
input1 = "1e-5pt"
result1 = resolver.size_to_pt(input1)
print(f"Input:  '{input1}'")
print(f"Result: '{result1}'")
print(f"Expected: '0.00001pt'")
print()

# Test case 2: Same value in decimal notation
input2 = "0.00001pt"
result2 = resolver.size_to_pt(input2)
print(f"Input:  '{input2}'")
print(f"Result: '{result2}'")
print(f"Expected: '0.00001pt'")
print()

# Test case 3: Specific failing value from property test
input3 = "6.103515625e-05pt"
result3 = resolver.size_to_pt(input3)
print(f"Input:  '{input3}'")
print(f"Result: '{result3}'")
print(f"Expected: '0.00006103515625pt' or '0.00006pt'")
print()

# Test case 4: Large value in scientific notation
input4 = "1e6pt"
result4 = resolver.size_to_pt(input4)
print(f"Input:  '{input4}'")
print(f"Result: '{result4}'")
print(f"Expected: '1000000pt'")
print()

# Test case 5: Scientific notation with decimal
input5 = "1.5e-3pt"
result5 = resolver.size_to_pt(input5)
print(f"Input:  '{input5}'")
print(f"Result: '{result5}'")
print(f"Expected: '0.0015pt'")