#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.computation.common import ensure_decoded

# Test with invalid UTF-8 bytes
invalid_utf8 = b'\x80'
print(f"Testing ensure_decoded with bytes: {invalid_utf8!r}")

try:
    result = ensure_decoded(invalid_utf8)
    print(f"Result: {result!r}")
    print(f"Result type: {type(result)}")
except Exception as e:
    print(f"Exception raised: {type(e).__name__}")
    print(f"Exception message: {e}")