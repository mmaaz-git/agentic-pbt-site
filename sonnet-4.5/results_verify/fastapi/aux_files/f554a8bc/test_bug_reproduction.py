"""Test to reproduce the reported bug in fastapi.utils.deep_dict_update"""

import copy
from hypothesis import given, strategies as st

from fastapi.utils import deep_dict_update

# First, let's test the basic reproduction example
def test_basic_reproduction():
    print("Testing basic reproduction example...")

    main = {"items": [1, 2]}
    update = {"items": [3]}

    print(f"Initial main: {main}")
    print(f"Update dict: {update}")

    deep_dict_update(main, update)
    print(f"After first update: {main}")

    deep_dict_update(main, update)
    print(f"After second update: {main}")

    print()

# Test with the simpler example from the bug report
def test_simple_example():
    print("Testing simple example from bug report...")

    main = {}
    update = {"items": [1, 2, 3]}

    print(f"Initial main: {main}")
    print(f"Update dict: {update}")

    deep_dict_update(main, update)
    first_result = copy.deepcopy(main)
    print(f"After first update: {first_result}")

    deep_dict_update(main, update)
    second_result = copy.deepcopy(main)
    print(f"After second update: {second_result}")

    if first_result == second_result:
        print("✓ Idempotence holds")
    else:
        print(f"✗ Idempotence violated!")
    print()

# Test with hypothesis
@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=3
    )
)
def test_deep_dict_update_idempotence_with_lists(update_dict):
    main_dict = {}

    deep_dict_update(main_dict, update_dict)
    first_result = copy.deepcopy(main_dict)

    deep_dict_update(main_dict, update_dict)
    second_result = copy.deepcopy(main_dict)

    assert first_result == second_result, (
        f"Idempotence violated: calling deep_dict_update twice with the same "
        f"update_dict produces different results. First: {first_result}, "
        f"Second: {second_result}"
    )

# Test how dicts behave (for comparison)
def test_dict_behavior():
    print("Testing dict behavior for comparison...")

    main = {"nested": {"key": "value1"}}
    update = {"nested": {"key": "value2"}}

    print(f"Initial main: {main}")
    print(f"Update dict: {update}")

    deep_dict_update(main, update)
    first_result = copy.deepcopy(main)
    print(f"After first update: {first_result}")

    deep_dict_update(main, update)
    second_result = copy.deepcopy(main)
    print(f"After second update: {second_result}")

    if first_result == second_result:
        print("✓ Idempotence holds for nested dicts")
    else:
        print(f"✗ Idempotence violated for nested dicts!")
    print()

# Test how simple values behave
def test_simple_value_behavior():
    print("Testing simple value behavior for comparison...")

    main = {"key": "value1"}
    update = {"key": "value2"}

    print(f"Initial main: {main}")
    print(f"Update dict: {update}")

    deep_dict_update(main, update)
    first_result = copy.deepcopy(main)
    print(f"After first update: {first_result}")

    deep_dict_update(main, update)
    second_result = copy.deepcopy(main)
    print(f"After second update: {second_result}")

    if first_result == second_result:
        print("✓ Idempotence holds for simple values")
    else:
        print(f"✗ Idempotence violated for simple values!")
    print()

if __name__ == "__main__":
    test_basic_reproduction()
    test_simple_example()
    test_dict_behavior()
    test_simple_value_behavior()

    # Run hypothesis test
    print("Running hypothesis-based property test...")
    try:
        test_deep_dict_update_idempotence_with_lists()
        print("✓ All hypothesis tests passed")
    except AssertionError as e:
        print(f"✗ Hypothesis test failed: {e}")