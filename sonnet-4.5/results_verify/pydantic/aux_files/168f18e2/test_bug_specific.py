import json
from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder

dec = Decimal('22367635711314.143')
encoded = decimal_encoder(dec)
json_str = json.dumps(encoded)
decoded = json.loads(json_str)
restored = Decimal(str(decoded))

print(f"Original: {dec}")
print(f"Encoded to: {encoded} (type: {type(encoded)})")
print(f"JSON string: {json_str}")
print(f"Decoded from JSON: {decoded} (type: {type(decoded)})")
print(f"After round-trip: {restored}")
print(f"Equal? {restored == dec}")
print(f"Difference: {restored - dec}")