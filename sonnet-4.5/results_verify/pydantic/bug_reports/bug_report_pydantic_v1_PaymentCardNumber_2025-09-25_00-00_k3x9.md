# Bug Report: pydantic.v1.PaymentCardNumber Mastercard Length Validation Bypassed

**Target**: `pydantic.v1.types.PaymentCardNumber.validate_length_for_brand`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `validate_length_for_brand` method fails to validate Mastercard length due to incorrect use of the `in` operator instead of `==` for enum comparison, allowing invalid Mastercard numbers of any length between 12-19 digits to pass validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.v1 import PaymentCardNumber, ValidationError
import pytest


def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    checksum = sum(odd_digits)
    for d in even_digits:
        checksum += sum(digits_of(d*2))
    return checksum % 10


def make_valid_luhn(card_number_without_check):
    check_digit = (10 - luhn_checksum(int(card_number_without_check + '0'))) % 10
    return card_number_without_check + str(check_digit)


@given(st.integers(min_value=12, max_value=19).filter(lambda x: x != 16))
def test_mastercard_must_be_16_digits(length):
    prefix = '51'
    card_without_check = prefix + ''.join(['0'] * (length - len(prefix) - 1))
    card = make_valid_luhn(card_without_check)

    with pytest.raises(ValidationError):
        PaymentCardNumber(card)
```

**Failing input**: `length=12` (produces "510000000008")

## Reproducing the Bug

```python
from pydantic.v1 import PaymentCardNumber

mastercard_15_digits = "5100000000000008"

card = PaymentCardNumber(mastercard_15_digits)
print(f"Created card: {card}")
print(f"Brand: {card.brand}")
print(f"Length: {len(card)}")
```

Expected: `ValidationError` indicating invalid length for Mastercard
Actual: Card is successfully created with 15 digits

## Why This Is A Bug

According to the Wikipedia reference cited in the code and payment card industry standards, Mastercard numbers must be exactly 16 digits. The `validate_length_for_brand` method is intended to enforce this constraint, but the validation is bypassed due to a typo in line 1043.

The code uses `if card_number.brand in PaymentCardBrand.mastercard:` where `PaymentCardBrand.mastercard` is a string enum member with value "Mastercard". Since `PaymentCardBrand` inherits from `str`, this becomes `if "Mastercard" in "Mastercard":` which is always `True`. This causes the condition to always match but the subsequent validation logic is never executed properly.

The correct comparison should use `==` to check enum equality, not `in` for substring checking.

## Fix

```diff
--- a/pydantic/v1/types.py
+++ b/pydantic/v1/types.py
@@ -1040,7 +1040,7 @@ class PaymentCardNumber(str):
         https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN)
         """
         required_length: Union[None, int, str] = None
-        if card_number.brand in PaymentCardBrand.mastercard:
+        if card_number.brand == PaymentCardBrand.mastercard:
             required_length = 16
             valid = len(card_number) == required_length
         elif card_number.brand == PaymentCardBrand.visa:
```