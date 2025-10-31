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

if __name__ == "__main__":
    # Test with specific example from the report
    test_mastercard_must_be_16_digits(12)
    print("Test passed with length=12")