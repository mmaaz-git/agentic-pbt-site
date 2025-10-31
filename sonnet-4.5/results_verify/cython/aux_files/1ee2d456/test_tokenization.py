#!/usr/bin/env python3
"""Test to understand tokenization of default statements"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import lex, parse

# Test tokenization
test_cases = [
    "{{default }}",
    "{{default}}",
    "{{default x}}",
    "{{default x=1}}",
]

for content in test_cases:
    print(f"\nTokenizing: {repr(content)}")
    try:
        tokens = lex(content)
        print(f"  Tokens: {tokens}")

        # Now try to parse
        print(f"  Parsing...")
        parsed = parse(tokens, 'test')
        print(f"  Parsed: {parsed}")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")