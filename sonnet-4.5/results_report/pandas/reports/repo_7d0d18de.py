import numpy.char as char
import numpy as np

print("=== Testing numpy.char.replace() truncation bug ===\n")

test_cases = [
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hello world'),
    ('test', 'test', 'testing'),
    ('Dr', 'Dr', 'Doctor'),
    ('foo', 'o', 'oo'),
    ('x', 'x', 'xyz'),
]

for haystack, old, new in test_cases:
    # Test with numpy.char.replace
    numpy_result = char.replace(haystack, old, new)
    numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)

    # Test with Python's str.replace
    python_result = haystack.replace(old, new)

    print(f"replace({repr(haystack)}, {repr(old)}, {repr(new)})")
    print(f"  numpy result:  {repr(numpy_str)}")
    print(f"  python result: {repr(python_result)}")
    print(f"  numpy dtype:   {numpy_result.dtype}")

    if numpy_str != python_result:
        print(f"  ❌ MISMATCH - Data truncated!")
    else:
        print(f"  ✓ Match")
    print()

print("\n=== Demonstrating the root cause ===\n")
print("The issue is that numpy.char.replace preserves the original dtype:")
print()

# Show how dtype affects the result
test_str = 'a'
for dtype_size in [1, 2, 5]:
    arr = np.array(test_str, dtype=f'<U{dtype_size}')
    result = char.replace(arr, 'a', 'aa')
    print(f"Input dtype: {arr.dtype}, value: {repr(str(arr))}")
    print(f"Output dtype: {result.dtype}, value: {repr(str(result))}")
    print(f"Expected: 'aa', Got: {repr(str(result))}, Correct: {str(result) == 'aa'}")
    print()