import numpy.char as char

test_cases = [
    ('ß', 'upper'),
    ('straße', 'upper'),
    ('ﬁ', 'upper'),
    ('ß', 'capitalize'),
    ('ß', 'swapcase'),
]

for s, func_name in test_cases:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item()
    python_result = python_func(s)

    print(f"{func_name}({repr(s)}):")
    print(f"  numpy:  {repr(numpy_result)} (length {len(numpy_result)})")
    print(f"  python: {repr(python_result)} (length {len(python_result)})")

    if numpy_result != python_result:
        print(f"  TRUNCATED!")
    print()