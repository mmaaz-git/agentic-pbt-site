import json
import warnings
from pydantic.deprecated.json import pydantic_encoder

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test case: bytes with non-UTF-8 sequences
non_utf8_bytes = b'\x80\x81\x82'

try:
    result = json.dumps(non_utf8_bytes, default=pydantic_encoder)
    print(f"Success: {result}")
except UnicodeDecodeError as e:
    print(f"UnicodeDecodeError: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")