from pydantic.v1 import PaymentCardNumber, ValidationError


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


# Test with specific lengths
for length in [12, 13, 14, 15, 17, 18, 19]:
    prefix = '51'
    card_without_check = prefix + ''.join(['0'] * (length - len(prefix) - 1))
    card = make_valid_luhn(card_without_check)

    print(f"\nTesting Mastercard with length {length}: {card}")
    try:
        card_obj = PaymentCardNumber(card)
        print(f"  ✗ Card created successfully (should have failed!)")
        print(f"  Brand: {card_obj.brand}, Length: {len(card_obj)}")
    except ValidationError as e:
        print(f"  ✓ ValidationError raised as expected: {e}")

# Also test the specific example from the bug report
print("\n" + "="*50)
print("Testing specific example from bug report:")
mastercard_15_digits = "5100000000000008"

try:
    card = PaymentCardNumber(mastercard_15_digits)
    print(f"Created card: {card}")
    print(f"Brand: {card.brand}")
    print(f"Length: {len(card)}")
    print("✗ Bug confirmed: Card with 15 digits was accepted!")
except ValidationError as e:
    print(f"✓ ValidationError raised: {e}")