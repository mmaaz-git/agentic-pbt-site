import copy
import matplotlib.units as munits
from hypothesis import given, settings, strategies as st, example
from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
import datetime


def test_register_deregister_inverse():
    """Test that single register/deregister cycle restores original state"""
    original_registry = copy.copy(munits.registry)

    # Show initial state
    had_datetime = datetime.datetime in original_registry
    print(f"  Initial: datetime.datetime in registry = {had_datetime}")

    register_matplotlib_converters()
    after_register = copy.copy(munits.registry)
    print(f"  After register: datetime.datetime in registry = {datetime.datetime in after_register}")

    deregister_matplotlib_converters()
    after_deregister = copy.copy(munits.registry)
    print(f"  After deregister: datetime.datetime in registry = {datetime.datetime in after_deregister}")

    # Check if registry was restored
    if original_registry != after_deregister:
        print(f"  Registry not restored! Original had {len(original_registry)} converters, now has {len(after_deregister)}")
        # Find differences
        for key in set(list(original_registry.keys()) + list(after_deregister.keys())):
            if key not in original_registry:
                print(f"    Added: {key}")
            elif key not in after_deregister:
                print(f"    Removed: {key}")
            elif original_registry[key] != after_deregister[key]:
                print(f"    Changed: {key}")

    assert original_registry == after_deregister, "Registry was not restored to original state"


@given(st.integers(min_value=1, max_value=5))
@example(1)  # Force testing with n=1
@example(2)  # Force testing with n=2
@settings(max_examples=10, deadline=None)
def test_register_deregister_multiple_times(n):
    """Test that multiple register/deregister cycles restore original state"""
    original_registry = copy.copy(munits.registry)

    # Show initial state
    had_datetime = datetime.datetime in original_registry
    print(f"  Testing with n={n}, initial datetime in registry = {had_datetime}")

    for i in range(n):
        register_matplotlib_converters()
        print(f"    After register #{i+1}: datetime.datetime in registry = {datetime.datetime in munits.registry}")

    for i in range(n):
        deregister_matplotlib_converters()
        print(f"    After deregister #{i+1}: datetime.datetime in registry = {datetime.datetime in munits.registry}")

    after_all = copy.copy(munits.registry)

    # Check if registry was restored
    if original_registry != after_all:
        print(f"  Registry not restored! Original had {len(original_registry)} converters, now has {len(after_all)}")

    assert original_registry == after_all, f"Registry was not restored after {n} cycles"


# Run the tests
print("Test 1: Single register/deregister cycle")
print("-" * 40)
try:
    test_register_deregister_inverse()
    print("PASSED")
except AssertionError as e:
    print(f"FAILED: {e}")

print("\nTest 2: Multiple register/deregister cycles with Hypothesis")
print("-" * 40)
try:
    test_register_deregister_multiple_times()
    print("ALL TESTS PASSED")
except AssertionError as e:
    print(f"FAILED: {e}")
except Exception as e:
    print(f"ERROR: {e}")