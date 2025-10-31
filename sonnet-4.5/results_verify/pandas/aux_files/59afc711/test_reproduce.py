import numpy.char as char

test_cases = [
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hello world'),
    ('test', 'test', 'testing'),
]

for haystack, old, new in test_cases:
    numpy_result = char.replace(haystack, old, new).item()
    python_result = haystack.replace(old, new)

    print(f"replace({repr(haystack)}, {repr(old)}, {repr(new)})")
    print(f"  numpy:  {repr(numpy_result)}")
    print(f"  python: {repr(python_result)}")

    if numpy_result != python_result:
        print(f"  MISMATCH!")