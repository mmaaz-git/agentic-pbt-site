import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import _tempita

# Trace exactly what happens with "{{def }}"
print("Tracing '{{def }}{{enddef}}':")

# First lex it
tokens = _tempita.lex("{{def }}{{enddef}}", 'test')
print(f"  Lexed tokens: {tokens}")

# The first token should be ('def ', position)
first_token, pos = tokens[0]
print(f"  First token: '{first_token}'")
print(f"  Starts with 'def ': {first_token.startswith('def ')}")

# Now trace what parse_expr does
print("\nWhat happens in parse_expr:")
print(f"  expr = '{first_token}'")
print(f"  expr.startswith('def '): {first_token.startswith('def ')}")

# If it starts with 'def ', it will call parse_def
if first_token.startswith('def '):
    print("  -> Will call parse_def()")
    print("\n  In parse_def:")
    print(f"    first = '{first_token}'")
    print(f"    Splitting: first.split(None, 1) = {first_token.split(None, 1)}")

    parts = first_token.split(None, 1)
    print(f"    Number of parts: {len(parts)}")
    if len(parts) < 2:
        print("    ERROR: Would crash with IndexError trying to access parts[1]!")
    else:
        print(f"    Would get: parts[1] = '{parts[1]}'")