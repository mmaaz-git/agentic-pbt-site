import copy
import matplotlib.units as munits
from hypothesis import given, settings, strategies as st
from pandas.plotting._matplotlib.converter import register, deregister


def test_register_deregister_inverse():
    original_registry = copy.copy(munits.registry)

    register()
    after_register = copy.copy(munits.registry)

    deregister()
    after_deregister = copy.copy(munits.registry)

    assert original_registry == after_deregister, f"Registry not restored properly\nOriginal: {original_registry}\nAfter: {after_deregister}"


@given(st.integers(min_value=1, max_value=5))
@settings(max_examples=50)
def test_register_deregister_multiple_times(n):
    original_registry = copy.copy(munits.registry)

    for _ in range(n):
        register()

    for _ in range(n):
        deregister()

    after_all = copy.copy(munits.registry)

    assert original_registry == after_all, f"Registry not restored after {n} cycles\nOriginal: {original_registry}\nAfter: {after_all}"


if __name__ == "__main__":
    print("Testing single register/deregister cycle...")
    try:
        test_register_deregister_inverse()
        print("✓ Single cycle test passed")
    except AssertionError as e:
        print(f"✗ Single cycle test failed: {e}")

    print("\nTesting multiple register/deregister cycles...")
    try:
        test_register_deregister_multiple_times()
        print("✓ Multiple cycles test passed")
    except AssertionError as e:
        print(f"✗ Multiple cycles test failed: {e}")