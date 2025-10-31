#!/usr/bin/env python3
"""Test the reported bug with decimal_encoder and special values."""

from decimal import Decimal
from hypothesis import given, strategies as st
import sys
import traceback

# Import the function
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')
from fastapi.encoders import decimal_encoder

print("Testing decimal_encoder with special values...")
print("=" * 60)

# Test 1: Regular decimals (should work)
print("\n1. Testing regular decimals:")
test_cases = [
    Decimal("1.0"),
    Decimal("1"),
    Decimal("-1.5"),
    Decimal("0"),
    Decimal("123.456"),
]

for dec in test_cases:
    try:
        result = decimal_encoder(dec)
        print(f"  decimal_encoder({dec!r}) = {result!r} (type: {type(result).__name__})")
    except Exception as e:
        print(f"  decimal_encoder({dec!r}) raised {type(e).__name__}: {e}")

# Test 2: Special values (the reported bug)
print("\n2. Testing special values:")
special_values = [
    Decimal('Infinity'),
    Decimal('-Infinity'),
    Decimal('NaN'),
]

for dec in special_values:
    try:
        result = decimal_encoder(dec)
        print(f"  decimal_encoder({dec!r}) = {result!r} (type: {type(result).__name__})")
    except Exception as e:
        print(f"  decimal_encoder({dec!r}) raised {type(e).__name__}: {e}")
        # Print traceback for the first error
        if dec == Decimal('Infinity'):
            print("\n  Full traceback:")
            traceback.print_exc()

# Test 3: Investigate as_tuple() behavior
print("\n3. Investigating as_tuple() for special values:")
for dec in [Decimal("1"), Decimal("1.5"), Decimal('Infinity'), Decimal('-Infinity'), Decimal('NaN')]:
    tuple_repr = dec.as_tuple()
    print(f"  {dec!r}.as_tuple() = {tuple_repr}")
    print(f"    exponent = {tuple_repr.exponent!r} (type: {type(tuple_repr.exponent).__name__})")

# Test 4: Run the hypothesis test from the bug report
print("\n4. Running the hypothesis test from the bug report:")

@st.composite
def decimal_with_special_values(draw):
    """Generate Decimal values including special values like Infinity and NaN."""
    choice = draw(st.integers(0, 9))
    if choice == 0:
        return Decimal('Infinity')
    elif choice == 1:
        return Decimal('-Infinity')
    elif choice == 2:
        return Decimal('NaN')
    else:
        return draw(st.decimals(allow_nan=False, allow_infinity=False))

@given(decimal_with_special_values())
def test_decimal_encoder_handles_all_decimal_values(dec_value):
    result = decimal_encoder(dec_value)
    assert isinstance(result, (int, float))

# Run a limited test
print("  Running hypothesis test (10 examples)...")
try:
    # Run with limited examples
    test_decimal_encoder_handles_all_decimal_values()
    print("  Test passed!")
except Exception as e:
    print(f"  Test failed with: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Bug reproduction complete.")