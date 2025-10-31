from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.typing import NAType


@given(st.lists(st.integers()) | st.dictionaries(st.integers(), st.integers()))
def test_na_equality_consistency_with_containers(container):
    result = pd.NA == container
    assert isinstance(result, NAType), (
        f"Expected NA == {type(container).__name__} to return NAType, "
        f"got {type(result).__name__}: {result}"
    )


def test_na_equality_consistency_with_none():
    result = pd.NA == None
    assert isinstance(result, NAType), (
        f"Expected NA == None to return NAType, got {type(result).__name__}: {result}"
    )

if __name__ == "__main__":
    print("Testing with Hypothesis:")
    print("=" * 50)

    # Test with None
    try:
        test_na_equality_consistency_with_none()
        print("✓ Test with None passed")
    except AssertionError as e:
        print(f"✗ Test with None failed: {e}")

    # Test with containers
    try:
        test_na_equality_consistency_with_containers([])
        print("✓ Test with empty list passed")
    except AssertionError as e:
        print(f"✗ Test with empty list failed: {e}")

    try:
        test_na_equality_consistency_with_containers([1, 2, 3])
        print("✓ Test with list [1, 2, 3] passed")
    except AssertionError as e:
        print(f"✗ Test with list [1, 2, 3] failed: {e}")

    try:
        test_na_equality_consistency_with_containers({})
        print("✓ Test with empty dict passed")
    except AssertionError as e:
        print(f"✗ Test with empty dict failed: {e}")

    try:
        test_na_equality_consistency_with_containers({1: 2, 3: 4})
        print("✓ Test with dict {1: 2, 3: 4} passed")
    except AssertionError as e:
        print(f"✗ Test with dict {1: 2, 3: 4} failed: {e}")