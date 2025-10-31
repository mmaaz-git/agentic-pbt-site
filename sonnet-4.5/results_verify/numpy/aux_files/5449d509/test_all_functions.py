import numpy as np
import numpy.char

# Test all mentioned functions with the problematic characters
test_chars = ['ẖ', 'ǰ', 'ß']
functions = [
    ('upper', numpy.char.upper, str.upper),
    ('swapcase', numpy.char.swapcase, str.swapcase),
    ('capitalize', numpy.char.capitalize, str.capitalize),
    ('title', numpy.char.title, str.title)
]

print("Testing all case transformation functions:")
print("="*60)

for func_name, numpy_func, python_func in functions:
    print(f"\nFunction: {func_name}")
    print("-"*40)
    for char in test_chars:
        arr = np.array([char])
        numpy_result = numpy_func(arr)[0]
        python_result = python_func(char)
        match = str(numpy_result) == python_result

        if not match:
            print(f"  '{char}' → NumPy: {repr(str(numpy_result))} (len={len(str(numpy_result))})")
            print(f"         Python: {repr(python_result)} (len={len(python_result)})")
            print(f"         Match: {match}")
        else:
            print(f"  '{char}' → OK (both return {repr(python_result)})")