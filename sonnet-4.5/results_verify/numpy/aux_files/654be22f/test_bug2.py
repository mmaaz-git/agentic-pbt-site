import numpy as np

# Test the specific case mentioned in the bug report
print("Testing specific case: number=0, padding=1")
result = np.base_repr(0, padding=1)
print(f"Result: '{result}'")
print(f"Expected (according to bug report): '00'")
print(f"Got: '{result}'")
print()

# Test other related cases for comparison
print("Testing related cases:")
print(f"np.base_repr(0, padding=0): '{np.base_repr(0, padding=0)}'")
print(f"np.base_repr(0, padding=1): '{np.base_repr(0, padding=1)}'")
print(f"np.base_repr(0, padding=2): '{np.base_repr(0, padding=2)}'")
print(f"np.base_repr(0, padding=3): '{np.base_repr(0, padding=3)}'")
print()

print(f"np.base_repr(1, padding=0): '{np.base_repr(1, padding=0)}'")
print(f"np.base_repr(1, padding=1): '{np.base_repr(1, padding=1)}'")
print(f"np.base_repr(1, padding=2): '{np.base_repr(1, padding=2)}'")
print()

print(f"np.base_repr(5, padding=0): '{np.base_repr(5, padding=0)}'")
print(f"np.base_repr(5, padding=1): '{np.base_repr(5, padding=1)}'")
print(f"np.base_repr(5, padding=2): '{np.base_repr(5, padding=2)}'")
print()

# Manual property test for failing case
def test_padding_property(number, base, padding):
    repr_with_padding = np.base_repr(number, base=base, padding=padding)
    repr_without_padding = np.base_repr(number, base=base, padding=0)
    expected_length = len(repr_without_padding) + padding
    actual_length = len(repr_with_padding)

    print(f"Testing number={number}, base={base}, padding={padding}")
    print(f"  Without padding: '{repr_without_padding}' (length={len(repr_without_padding)})")
    print(f"  With padding={padding}: '{repr_with_padding}' (length={actual_length})")
    print(f"  Expected length: {expected_length}")
    print(f"  Test {'PASSED' if actual_length == expected_length else 'FAILED'}")
    print()

    return actual_length == expected_length

print("\nProperty test on failing input:")
test_padding_property(0, 2, 1)

print("Property test on other inputs:")
test_padding_property(0, 2, 2)
test_padding_property(1, 2, 1)
test_padding_property(5, 2, 1)