import numpy.char as char

print("Testing swapcase with expanding characters:")

test_cases = [
    ('ß', 'swapcase'),  # lowercase should become uppercase SS
    ('ﬁle', 'swapcase'),  # ligature should expand
]

for s, func_name in test_cases:
    numpy_func = getattr(char, func_name)
    python_func = getattr(str, func_name)

    numpy_result = numpy_func(s).item() if hasattr(numpy_func(s), 'item') else str(numpy_func(s))
    python_result = python_func(s)

    print(f"\n{func_name}({repr(s)}):")
    print(f"  numpy:  {repr(numpy_result)} (length {len(numpy_result)})")
    print(f"  python: {repr(python_result)} (length {len(python_result)})")

    if numpy_result != python_result:
        print(f"  MISMATCH!")