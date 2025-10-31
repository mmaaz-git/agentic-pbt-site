#!/usr/bin/env python3
"""Test to reproduce the reported bug with Django UserSettingsHolder"""

from django.conf import UserSettingsHolder, global_settings, LazySettings


# Direct reproduction
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


# Test specific case
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


# Test with getattr and direct access
def test_various_access_methods():
    """Test different ways of accessing settings"""
    print("\n=== Testing Various Access Methods ===")

    holder = UserSettingsHolder(global_settings)

    # Set lowercase via direct assignment
    holder.lowercase_setting = "value1"

    # Set uppercase via direct assignment
    holder.UPPERCASE_SETTING = "value2"

    print(f"Direct access to lowercase: holder.lowercase_setting = {holder.lowercase_setting}")
    print(f"Direct access to uppercase: holder.UPPERCASE_SETTING = {holder.UPPERCASE_SETTING}")

    # Try getattr on lowercase
    try:
        val = getattr(holder, "lowercase_setting")
        print(f"getattr(holder, 'lowercase_setting') = {val}")
    except AttributeError as e:
        print(f"getattr(holder, 'lowercase_setting') raised AttributeError: {e}")

    # Check what __getattr__ does
    print("\n--- Testing __getattr__ directly ---")
    try:
        val = holder.__getattr__("lowercase_setting")
        print(f"holder.__getattr__('lowercase_setting') = {val}")
    except AttributeError:
        print("holder.__getattr__('lowercase_setting') raised AttributeError (expected)")

    # Check __dict__
    print("\n--- Checking __dict__ ---")
    print(f"'lowercase_setting' in holder.__dict__: {'lowercase_setting' in holder.__dict__}")
    print(f"'UPPERCASE_SETTING' in holder.__dict__: {'UPPERCASE_SETTING' in holder.__dict__}")


if __name__ == "__main__":
    test_direct_reproduction()
    test_specific_case()
    test_various_access_methods()