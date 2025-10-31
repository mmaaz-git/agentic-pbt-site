#!/usr/bin/env python3
"""Run the EXACT code from the bug report's reproduction section"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

import llm

values = [1.0, 2.0, 3.0]
encoded = llm.encode(values)

corrupted_bytes = encoded + b'\x00'

try:
    decoded = llm.decode(corrupted_bytes)

    print(f"Input length: {len(corrupted_bytes)} (not multiple of 4)")
    print(f"Decoded: {list(decoded)}")
    print(f"Lost data: {len(corrupted_bytes) - len(decoded) * 4} bytes silently discarded")
    print("\nBUG REPORT IS CORRECT: Silent truncation occurred!")
except Exception as e:
    print(f"Input length: {len(corrupted_bytes)} (not multiple of 4)")
    print(f"Exception raised: {type(e).__name__}: {e}")
    print("\nBUG REPORT IS INCORRECT: An exception was raised, not silent truncation")