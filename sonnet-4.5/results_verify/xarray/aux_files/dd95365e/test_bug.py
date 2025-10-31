#!/usr/bin/env python3

# Test 1: Run the Hypothesis test
from hypothesis import given, strategies as st, settings
from xarray.core.formatting_html import collapsible_section

@given(st.integers(min_value=-100, max_value=-1))
@settings(max_examples=10)
def test_collapsible_section_negative_n_items_should_be_disabled(n_items):
    result = collapsible_section("Test", "", "", n_items=n_items, enabled=True, collapsed=False)

    print(f"\nTesting with n_items={n_items}")
    print(f"Result contains 'disabled': {'disabled' in result}")
    print(f"Result contains 'checked': {'checked' in result}")

    assert "disabled" in result, \
        f"Negative n_items={n_items} should result in disabled section"
    assert "checked" not in result, \
        f"Negative n_items={n_items} should not result in checked/expanded section"

print("=" * 60)
print("Running Hypothesis test:")
print("=" * 60)
try:
    test_collapsible_section_negative_n_items_should_be_disabled()
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Test 2: Manual reproduction with n_items=-5
print("\n" + "=" * 60)
print("Manual test with n_items=-5:")
print("=" * 60)

result = collapsible_section(
    name="Test Section",
    n_items=-5,
    enabled=True,
    collapsed=False
)

print(f"\nResult HTML:\n{result}")
print("\n" + "-" * 40)

# Simulate the logic
has_items = -5 is not None and -5
print(f"has_items evaluates to: {has_items}")
print(f"bool(has_items) = {bool(has_items)}")
print(f"'disabled' in result: {'disabled' in result}")
print(f"'checked' in result: {'checked' in result}")

# Test 3: Compare different values
print("\n" + "=" * 60)
print("Comparing different n_items values:")
print("=" * 60)

test_values = [None, 0, -1, -5, 1, 5]
for val in test_values:
    result = collapsible_section("Test", "", "", n_items=val, enabled=True, collapsed=False)
    has_disabled = "disabled" in result
    has_checked = "checked" in result
    val_str = "None" if val is None else str(val)
    print(f"n_items={val_str:>4}: disabled={has_disabled}, checked={has_checked}")