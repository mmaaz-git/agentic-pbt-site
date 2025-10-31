from decimal import Decimal
from pydantic.deprecated.json import decimal_encoder
import warnings

warnings.filterwarnings("ignore")

x = Decimal('252579977670696.67')
encoded = decimal_encoder(x)
decoded = Decimal(str(encoded))

print(f"Original: {x}")
print(f"Encoded:  {encoded}")
print(f"Decoded:  {decoded}")
print(f"Round-trip equal? {decoded == x}")