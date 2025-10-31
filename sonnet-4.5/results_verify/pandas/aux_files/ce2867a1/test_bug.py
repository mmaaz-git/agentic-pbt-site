import matplotlib.units as munits
import pandas.plotting

# Test 1: Basic reproduction
print("=== Test 1: Basic reproduction ===")
initial_keys = set(munits.registry.keys())
print(f"Initial registry has {len(initial_keys)} converter(s):")
for key in sorted([str(k) for k in initial_keys]):
    print(f"  {key}")

pandas.plotting.register_matplotlib_converters()
after_register_keys = set(munits.registry.keys())
print(f"\nAfter register: {len(after_register_keys)} converters")

pandas.plotting.deregister_matplotlib_converters()
after_deregister_keys = set(munits.registry.keys())
print(f"After deregister: {len(after_deregister_keys)} converters")

extra = after_deregister_keys - initial_keys
print(f"\nExtra converters ({len(extra)}):")
for key in sorted([str(k) for k in extra]):
    print(f"  {key}")

print("\n=== Test 2: Property-based test ===")
import pytest

def test_deregister_restores_original_state():
    """
    Property: deregister_matplotlib_converters should restore the matplotlib
    registry to its original state before register was called.

    This is explicitly stated in the docstring: "This attempts to set the
    state of the registry back to the state before pandas registered its
    own units."
    """
    # Reset registry
    munits.registry.clear()
    munits.registry[type(1.0)] = munits.registry.get(type(1.0))  # Add back default

    initial_keys = set(munits.registry.keys())

    pandas.plotting.register_matplotlib_converters()
    pandas.plotting.deregister_matplotlib_converters()

    after_deregister_keys = set(munits.registry.keys())

    assert after_deregister_keys == initial_keys, (
        f"deregister() should restore the original registry state, but "
        f"initial keys: {sorted([str(k) for k in initial_keys])}, "
        f"after deregister keys: {sorted([str(k) for k in after_deregister_keys])}"
    )

try:
    test_deregister_restores_original_state()
    print("Property test PASSED")
except AssertionError as e:
    print(f"Property test FAILED: {e}")

print("\n=== Test 3: Multiple register/deregister calls ===")
# Clear registry first
munits.registry.clear()
initial = set(munits.registry.keys())
print(f"Initial registry (cleared): {len(initial)} converters")

# First round
pandas.plotting.register_matplotlib_converters()
print(f"After 1st register: {len(munits.registry)} converters")
pandas.plotting.deregister_matplotlib_converters()
print(f"After 1st deregister: {len(munits.registry)} converters")

# Second round
pandas.plotting.register_matplotlib_converters()
print(f"After 2nd register: {len(munits.registry)} converters")
pandas.plotting.deregister_matplotlib_converters()
print(f"After 2nd deregister: {len(munits.registry)} converters")