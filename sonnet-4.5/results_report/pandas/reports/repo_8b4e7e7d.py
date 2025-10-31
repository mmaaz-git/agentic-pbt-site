from pandas.io.formats.css import CSSResolver

resolver = CSSResolver()

# Test with direct pt values
result = resolver("font-size: 1.5pt")
print(f"Input: 'font-size: 1.5pt' -> Output: {result.get('font-size', 'None')}")

# Test with px values that convert to pt
result = resolver("margin: 10px")
print(f"Input: 'margin: 10px' -> Output (margin-top): {result.get('margin-top', 'None')}")

# Test with different decimal values
result = resolver("font-size: 3.75pt")
print(f"Input: 'font-size: 3.75pt' -> Output: {result.get('font-size', 'None')}")

# Test with integer pt value (should not have trailing zeros)
result = resolver("font-size: 5pt")
print(f"Input: 'font-size: 5pt' -> Output: {result.get('font-size', 'None')}")

# Test with small decimal value
result = resolver("font-size: 0.5pt")
print(f"Input: 'font-size: 0.5pt' -> Output: {result.get('font-size', 'None')}")

# Test with value that rounds
result = resolver("font-size: 1.3333333pt")
print(f"Input: 'font-size: 1.3333333pt' -> Output: {result.get('font-size', 'None')}")