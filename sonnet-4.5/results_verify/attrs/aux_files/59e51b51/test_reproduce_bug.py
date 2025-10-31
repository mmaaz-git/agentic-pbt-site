#!/usr/bin/env python3
"""Test script to reproduce the or_() validator exception bug"""

import attrs
from attrs import validators

print("=" * 60)
print("Testing exception format consistency between validators")
print("=" * 60)

# Test 1: in_() validator exception format
print("\n1. Testing in_() validator exception format:")
@attrs.define
class Example:
    value: int = attrs.field(validator=validators.in_([1, 2, 3]))

try:
    Example(value=999)
except ValueError as e:
    print(f"   in_() exception args count: {len(e.args)}")
    print(f"   Args: {e.args}")
    if len(e.args) >= 2:
        print(f"   - Message: {e.args[0]}")
        print(f"   - Attribute type: {type(e.args[1])}")
        if len(e.args) >= 3:
            print(f"   - Options: {e.args[2]}")
        if len(e.args) >= 4:
            print(f"   - Value: {e.args[3]}")

# Test 2: or_() validator exception format
print("\n2. Testing or_() validator exception format:")
@attrs.define
class Example2:
    value: int = attrs.field(validator=validators.or_(
        validators.instance_of(str),
        validators.instance_of(list)
    ))

try:
    Example2(value=999)
except (ValueError, TypeError) as e:
    print(f"   or_() exception args count: {len(e.args)}")
    print(f"   Args: {e.args}")
    if len(e.args) >= 2:
        print(f"   - Message: {e.args[0]}")
        print(f"   - Attribute type: {type(e.args[1])}")

# Test 3: instance_of() validator exception format
print("\n3. Testing instance_of() validator exception format:")
@attrs.define
class Example3:
    value: int = attrs.field(validator=validators.instance_of(str))

try:
    Example3(value=999)
except (ValueError, TypeError) as e:
    print(f"   instance_of() exception args count: {len(e.args)}")
    print(f"   Args: {e.args}")
    if len(e.args) >= 2:
        print(f"   - Message: {e.args[0]}")
        print(f"   - Attribute type: {type(e.args[1])}")

# Test 4: Property-based test from bug report
print("\n4. Running property-based test:")
from hypothesis import given, strategies as st

@given(st.integers())
def test_or_validator_exception_format_consistency(value):
    @attrs.define
    class Example:
        num: int = attrs.field(validator=validators.or_(
            validators.instance_of(str),
            validators.instance_of(list)
        ))

    try:
        Example(num=value)
    except (ValueError, TypeError) as e:
        assert len(e.args) >= 2, \
            f"or_() should pass (msg, attr, ...) but only passed {len(e.args)} args: {e.args}"

try:
    test_or_validator_exception_format_consistency()
    print("   Property test PASSED (should have failed according to bug report)")
except AssertionError as e:
    print(f"   Property test FAILED as expected: {e}")
except Exception as e:
    print(f"   Property test raised unexpected error: {e}")

print("\n" + "=" * 60)
print("Summary:")
print("If or_() only passes 1 arg while other validators pass multiple,")
print("this confirms the inconsistency reported in the bug.")