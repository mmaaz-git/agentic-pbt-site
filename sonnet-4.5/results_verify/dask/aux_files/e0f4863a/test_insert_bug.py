#!/usr/bin/env python3
"""Test the dask.utils.insert function bug report."""

from hypothesis import given, strategies as st, settings
from dask.utils import insert


@settings(max_examples=500)
@given(
    st.tuples(st.integers(), st.integers(), st.integers()),
    st.integers(min_value=0, max_value=2),
    st.integers()
)
def test_insert_preserves_length_not_increases(tup, loc, val):
    """
    BUG: insert() preserves tuple length instead of increasing it.

    A true insertion should increase the length by 1.
    But dask.utils.insert() replaces an element, keeping the same length.
    """
    result = insert(tup, loc, val)

    # BUG: length is preserved (replacement behavior)
    assert len(result) == len(tup)

    # Element at loc is replaced with val
    assert result[loc] == val

    # All other elements unchanged
    for i in range(len(tup)):
        if i != loc:
            assert result[i] == tup[i]


def reproduce_manual():
    """Reproduce the bug manually."""
    print("=== Manual Reproduction ===")

    tup = ('a', 'b', 'c')
    result = insert(tup, 1, 'X')
    print(f"Input tuple: {tup}")
    print(f"insert(tup, 1, 'X') result: {result}")

    print(f"\nLength before: {len(tup)}")
    print(f"Length after: {len(result)}")
    print(f"Element at position 1: {result[1]}")

    # Show what Python's list.insert would do
    lst = list(tup)
    lst.insert(1, 'X')
    print(f"\nTrue insert (list.insert) would give: {tuple(lst)}")

    # Test the docstring example
    print("\n=== Docstring Example ===")
    docstring_result = insert(('a', 'b', 'c'), 0, 'x')
    print(f"insert(('a', 'b', 'c'), 0, 'x') = {docstring_result}")
    print(f"This shows REPLACEMENT, not insertion")

    # What true insertion would look like
    true_insert = ['a', 'b', 'c']
    true_insert.insert(0, 'x')
    print(f"True insertion would be: {tuple(true_insert)}")


if __name__ == "__main__":
    print("Running Hypothesis tests...")
    try:
        test_insert_preserves_length_not_increases()
        print("✓ Hypothesis tests passed (500 examples)")
        print("  - Function preserves tuple length (replacement behavior)")
        print("  - Function replaces element at specified index")
        print("  - Other elements remain unchanged")
    except Exception as e:
        print(f"✗ Hypothesis tests failed: {e}")

    print("\n" + "="*50 + "\n")
    reproduce_manual()