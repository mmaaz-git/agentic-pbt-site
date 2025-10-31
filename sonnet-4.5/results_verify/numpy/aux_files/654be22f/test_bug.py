import numpy as np
from hypothesis import given, strategies as st

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

# Test the property-based test
print("Running property-based test on failing input:")
@given(st.integers(min_value=0, max_value=10000), st.integers(min_value=2, max_value=36), st.integers(min_value=1, max_value=20))
def test_base_repr_padding_adds_exact_zeros(number, base, padding):
    repr_with_padding = np.base_repr(number, base=base, padding=padding)
    repr_without_padding = np.base_repr(number, base=base, padding=0)
    expected_length = len(repr_without_padding) + padding
    assert len(repr_with_padding) == expected_length, f"Failed for number={number}, base={base}, padding={padding}. Expected length {expected_length}, got {len(repr_with_padding)}"

# Test the specific failing case
try:
    test_base_repr_padding_adds_exact_zeros(0, 2, 1)
    print("Property test passed for (0, 2, 1)")
except AssertionError as e:
    print(f"Property test failed for (0, 2, 1): {e}")