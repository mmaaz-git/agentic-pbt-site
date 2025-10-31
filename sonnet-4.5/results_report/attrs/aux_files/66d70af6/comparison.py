#!/usr/bin/env python3
"""Comparison of exception handling in different validators"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr


class BuggyValidator:
    """A validator with a programming error (NameError)"""
    def __call__(self, inst, attr, value):
        undefined_variable  # This will raise NameError


# Create test attribute
field_attr = attr.Attribute(
    name='test',
    default=attr.NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    eq=True,
    eq_key=None,
    order=False,
    order_key=None,
    hash=None,
    init=True,
    metadata={},
    type=None,
    converter=None,
    kw_only=False,
    inherited=False,
    on_setattr=None,
    alias=None
)

print("Testing exception handling in different validators:\n")

# Test 1: and_ validator
print("1. and_() validator with buggy validator:")
and_validator = attr.validators.and_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)
try:
    and_validator(None, field_attr, 42)
    print("   No error raised")
except NameError as e:
    print(f"   NameError propagated correctly: {e}")
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

# Test 2: not_ validator
print("\n2. not_() validator with buggy validator:")
not_validator = attr.validators.not_(BuggyValidator())
try:
    not_validator(None, field_attr, 42)
    print("   No error raised")
except NameError as e:
    print(f"   NameError propagated correctly: {e}")
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")

# Test 3: or_ validator
print("\n3. or_() validator with buggy validator:")
or_validator = attr.validators.or_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)
try:
    or_validator(None, field_attr, 42)
    print("   No error raised - BUG! NameError was silently caught!")
except NameError as e:
    print(f"   NameError propagated correctly: {e}")
except Exception as e:
    print(f"   Other error: {type(e).__name__}: {e}")