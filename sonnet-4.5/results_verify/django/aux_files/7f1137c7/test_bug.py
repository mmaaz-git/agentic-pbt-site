import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.signing import b62_encode, b62_decode

print("Bug 1: Crash on empty string")
try:
    result = b62_decode("")
    print(f"Result: {result}")
except IndexError as e:
    print(f"CRASH: IndexError - {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__} - {e}")

print("\nBug 2: Invalid round-trip for '-'")
try:
    result = b62_decode("-")
    print(f"b62_decode('-') = {result}")
    reencoded = b62_encode(result)
    print(f"b62_encode({result}) = '{reencoded}'")
    print(f"Round-trip: '-' -> {result} -> '{reencoded}'")
    print(f"Expected: '-' should either raise an error or round-trip correctly")
except Exception as e:
    print(f"Exception: {type(e).__name__} - {e}")

print("\nBug 3: Impact on TimestampSigner")
from django.core.signing import TimestampSigner
from django.conf import settings
if not settings.configured:
    settings.configure(SECRET_KEY='test', SECRET_KEY_FALLBACKS=[])

signer = TimestampSigner()
signed = signer.sign("test")
print(f"Signed value: {signed}")
parts = signed.split(':')
malformed = f"{parts[0]}::{parts[2]}" if len(parts) >= 3 else "test::"
print(f"Malformed signature: {malformed}")

try:
    result = signer.unsign(malformed)
    print(f"Unsigned result: {result}")
except IndexError as e:
    print("CRASH: TimestampSigner.unsign() crashes on malformed signature with empty timestamp")
    print(f"IndexError: {e}")
except Exception as e:
    print(f"Handled: {type(e).__name__} - {e}")
