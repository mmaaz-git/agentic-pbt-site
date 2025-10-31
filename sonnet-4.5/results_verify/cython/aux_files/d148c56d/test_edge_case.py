#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import parse, lex

# Check what tokens are produced by lex for various inputs
test_cases = [
    "{{default }}",
    "{{default x=1}}",
    "{{default  }}",  # two spaces after default
    "{{default\t}}",
    "{{ default }}",  # space before default
]

for test in test_cases:
    print(f"\nTesting: {repr(test)}")
    tokens = lex(test)
    print(f"  Tokens: {tokens}")

    # Check what parse_expr would do with this token
    if tokens:
        token = tokens[0]
        if isinstance(token, tuple):
            expr, pos = token
            expr_stripped = expr.strip()
            print(f"  Token expression: {repr(expr)}")
            print(f"  Stripped: {repr(expr_stripped)}")
            print(f"  Starts with 'default ': {expr_stripped.startswith('default ')}")

            # If it starts with 'default ', what would split do?
            if expr_stripped.startswith('default '):
                parts = expr_stripped.split(None, 1)
                print(f"  Split result: {parts}")
                if len(parts) >= 2:
                    print(f"  Would get: {repr(parts[1])}")
                else:
                    print(f"  Would cause IndexError!")

    # Now try to parse it
    try:
        result = parse(test)
        print(f"  Parse result: {result}")
    except Exception as e:
        print(f"  Parse error: {type(e).__name__}: {e}")