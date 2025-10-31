#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.conf import settings
settings.configure(USE_I18N=False)

from decimal import Decimal
from django.core.validators import DecimalValidator
from django.core.exceptions import ValidationError

validator = DecimalValidator(max_digits=1, decimal_places=0)

print("Testing DecimalValidator(max_digits=1, decimal_places=0)")
print("=" * 50)

print("Decimal('0'):", end=" ")
try:
    validator(Decimal("0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - {e}")

print("Decimal('0.0'):", end=" ")
try:
    validator(Decimal("0.0"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - BUG - {e}")

print("\nAdditional tests:")
print("Decimal('0.00'):", end=" ")
try:
    validator(Decimal("0.00"))
    print("PASSED ✓")
except ValidationError as e:
    print(f"FAILED ✗ - {e}")

print("\nTesting values of zero:")
print(f"Decimal('0') == Decimal('0.0'): {Decimal('0') == Decimal('0.0')}")
print(f"Decimal('0') == Decimal('0.00'): {Decimal('0') == Decimal('0.00')}")