import json
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

def test_round_trip(original_value, context=""):
    """Test if a value can round-trip through JSON encoding/decoding."""
    print(f"\nTesting round-trip for {original_value} {context}:")
    print(f"  Original: {original_value} (type: {type(original_value).__name__})")

    # Encode using decimal_encoder
    encoded = decimal_encoder(original_value)
    print(f"  Encoded: {encoded} (type: {type(encoded).__name__})")

    # Convert to JSON string
    json_str = json.dumps(encoded)
    print(f"  JSON string: {json_str}")

    # Parse back from JSON
    parsed = json.loads(json_str)
    print(f"  Parsed: {parsed} (type: {type(parsed).__name__})")

    # Try to convert back to Decimal
    restored = Decimal(str(parsed))
    print(f"  Restored: {restored} (type: {type(restored).__name__})")

    # Check if round-trip preserved the value
    success = original_value == restored
    print(f"  Round-trip successful: {success}")

    if not success:
        print(f"  ERROR: Value changed from {original_value} to {restored}")

    return success

print("=" * 60)
print("Testing round-trip behavior with ConstrainedDecimal scenarios")
print("(simulating Numeric(x,0) - integers stored as decimals)")
print("=" * 60)

# Test cases that would come from a database Numeric(10,0) field
test_values = [
    (Decimal('1'), "from Numeric(10,0)"),
    (Decimal('1.0'), "from Numeric(10,0) with trailing zero"),
    (Decimal('42'), "from Numeric(10,0)"),
    (Decimal('42.00'), "from Numeric(10,0) with trailing zeros"),
    (Decimal('123456789'), "from Numeric(10,0)"),
]

all_success = True
for val, context in test_values:
    success = test_round_trip(val, context)
    all_success = all_success and success

print("\n" + "=" * 60)
print(f"Overall round-trip success: {all_success}")

# Now let's see what happens if we use the proposed fix
print("\n" + "=" * 60)
print("Testing with proposed fix logic:")
print("=" * 60)

def decimal_encoder_fixed(dec_value: Decimal):
    """Proposed fix: check if value is an integer regardless of representation."""
    if dec_value == dec_value.to_integral_value():
        return int(dec_value)
    else:
        return float(dec_value)

for val, context in test_values:
    print(f"\n{val} {context}:")
    encoded_current = decimal_encoder(val)
    encoded_fixed = decimal_encoder_fixed(val)
    print(f"  Current encoding: {encoded_current} (type: {type(encoded_current).__name__})")
    print(f"  Fixed encoding: {encoded_fixed} (type: {type(encoded_fixed).__name__})")
    print(f"  Are they different? {encoded_current != encoded_fixed}")