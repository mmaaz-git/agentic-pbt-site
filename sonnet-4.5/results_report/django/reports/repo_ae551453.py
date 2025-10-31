import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.signing import b62_encode, b62_decode

print("Bug 1: Crash on empty string")
try:
    result = b62_decode("")
    print(f"Result: {result}")
except IndexError as e:
    print(f"CRASH: IndexError - {e}")

print("\nBug 2: Invalid round-trip for '-'")
result = b62_decode("-")
print(f"b62_decode('-') = {result}")
reencoded = b62_encode(result)
print(f"b62_encode({result}) = '{reencoded}'")
print(f"Round-trip: '-' -> {result} -> '{reencoded}'")
print(f"Expected: '-' should either raise an error or round-trip correctly")

print("\nBug 3: Impact on TimestampSigner")
from django.core.signing import TimestampSigner
from django.conf import settings
if not settings.configured:
    settings.configure(SECRET_KEY='test', SECRET_KEY_FALLBACKS=[])

signer = TimestampSigner()
signed = signer.sign("test")
parts = signed.split(':')
malformed = f"{parts[0]}::{parts[2]}"

try:
    result = signer.unsign(malformed)
except IndexError:
    print("CRASH: TimestampSigner.unsign() crashes on malformed signature with empty timestamp")
except Exception as e:
    print(f"Handled: {type(e).__name__}")