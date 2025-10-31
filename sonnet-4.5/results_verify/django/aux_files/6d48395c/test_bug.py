#!/usr/bin/env python3
"""Test to reproduce the reported bug with Django UserSettingsHolder"""

from hypothesis import given, strategies as st, assume
from django.conf import UserSettingsHolder, global_settings, LazySettings


# First test: The hypothesis test
@given(st.text(min_size=1))
def test_usersettingsholder_uppercase_contract(setting_name):
    """
    UserSettingsHolder should enforce that settings are uppercase,
    consistent with LazySettings.configure()'s validation.
    """
    assume(setting_name.upper() != setting_name)
    assume(setting_name not in {'default_settings', 'SETTINGS_MODULE'})
    assume(not setting_name.startswith('_'))

    holder = UserSettingsHolder(global_settings)

    holder.__setattr__(setting_name, "test_value")

    try:
        holder.__getattribute__(setting_name)
        assert False, f"Should not allow setting/getting lowercase setting {setting_name!r}"
    except AttributeError:
        pass


# Second test: Direct reproduction
def test_direct_reproduction():
    """Direct test reproduction from the bug report"""
    print("\n=== Direct Reproduction Test ===")

    holder = UserSettingsHolder(global_settings)
    holder.my_setting = "test"
    print(f"holder.my_setting = {holder.my_setting}")

    settings = LazySettings()
    try:
        settings.configure(my_setting="test")
        print("configure() accepted lowercase setting")
    except TypeError as e:
        print(f"configure() rejects lowercase: {e}")


# Run specific test case
def test_specific_case():
    """Test the specific failing case mentioned in report"""
    print("\n=== Specific Test Case: setting_name='a' ===")

    setting_name = 'a'
    holder = UserSettingsHolder(global_settings)

    # Set the lowercase attribute
    holder.__setattr__(setting_name, "test_value")

    # Try to get it
    try:
        value = holder.__getattribute__(setting_name)
        print(f"Successfully set and retrieved lowercase setting '{setting_name}': {value}")
        print("This demonstrates the bug - lowercase settings are allowed")
    except AttributeError as e:
        print(f"AttributeError when accessing '{setting_name}': {e}")


if __name__ == "__main__":
    # Run direct reproduction
    test_direct_reproduction()

    # Run specific test case
    test_specific_case()

    # Run hypothesis test with specific example
    print("\n=== Hypothesis Test with setting_name='a' ===")
    try:
        test_usersettingsholder_uppercase_contract('a')
        print("Hypothesis test passed (no assertion error)")
    except AssertionError as e:
        print(f"Hypothesis test failed with assertion: {e}")
    except Exception as e:
        print(f"Hypothesis test failed with exception: {e}")