#!/usr/bin/env python3
"""Minimal reproduction case for attr.validators.or_ bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/attrs_env/lib/python3.13/site-packages')

import attr


class BuggyValidator:
    """A validator with a programming error (NameError)"""
    def __call__(self, inst, attr, value):
        # This will raise NameError when called
        undefined_variable


# Create an or_ validator that includes our buggy validator
validator = attr.validators.or_(
    BuggyValidator(),
    attr.validators.instance_of(int)
)

# Create a minimal Attribute object for testing
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

# Test with integer value - this SHOULD raise NameError but doesn't
print("Testing or_ validator with buggy validator that has NameError...")
print(f"Input value: 42")
print(f"Expected: NameError: name 'undefined_variable' is not defined")
print(f"Actual: ", end="")

try:
    validator(None, field_attr, 42)
    print("No error raised! The NameError was silently caught and the second validator passed.")
except NameError as e:
    print(f"NameError raised (correct): {e}")
except Exception as e:
    print(f"Different error raised: {type(e).__name__}: {e}")