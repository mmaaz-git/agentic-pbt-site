import numpy as np
import numpy.char

# Test character that demonstrates the bug
test_char = 'ẖ'
arr = np.array([test_char])

# Get results from NumPy and Python
numpy_result = numpy.char.swapcase(arr)[0]
python_result = test_char.swapcase()

print(f"Input: '{test_char}' (U+{ord(test_char):04X}, LATIN SMALL LETTER H WITH LINE BELOW)")
print(f"Python str.swapcase: '{python_result}' (len={len(python_result)})")
print(f"NumPy char.swapcase: '{numpy_result}' (len={len(numpy_result)})")
print(f"Match: {str(numpy_result) == python_result}")

# Show the character codes to demonstrate truncation
print(f"\nPython result characters: {[f'U+{ord(c):04X}' for c in python_result]}")
print(f"NumPy result characters: {[f'U+{ord(c):04X}' for c in str(numpy_result)]}")

print("\n--- Testing other affected functions and characters ---")

# Test multiple functions with different problematic characters
test_cases = [
    ('ẖ', 'LATIN SMALL LETTER H WITH LINE BELOW'),
    ('ǰ', 'LATIN SMALL LETTER J WITH CARON'),
    ('ß', 'LATIN SMALL LETTER SHARP S (German eszett)')
]

functions = [
    ('upper', numpy.char.upper),
    ('swapcase', numpy.char.swapcase),
    ('capitalize', numpy.char.capitalize),
    ('title', numpy.char.title)
]

for char, description in test_cases:
    print(f"\nCharacter: '{char}' (U+{ord(char):04X}, {description})")
    for func_name, numpy_func in functions:
        arr = np.array([char])
        numpy_res = str(numpy_func(arr)[0])
        python_res = getattr(char, func_name)()
        match = "✓" if numpy_res == python_res else "✗"
        print(f"  {func_name:10} → NumPy: {repr(numpy_res):8} Python: {repr(python_res):8} {match}")