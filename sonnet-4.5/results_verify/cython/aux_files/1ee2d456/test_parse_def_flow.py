import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Tempita._tempita as tempita

original_parse_def = tempita.parse_def

def traced_parse_def(tokens, name, context):
    print(f"\nparse_def called!")
    first, start = tokens[0]
    print(f"  first token: '{first}'")
    tokens = tokens[1:]

    assert first.startswith('def ')

    # This is the problematic line
    parts = first.split(None, 1)
    print(f"  split result: {parts}")
    if len(parts) < 2:
        print(f"  ERROR: No function name, parts[1] would crash!")
        # Let's see what happens if we provide empty string as function name
        first_after = ''
    else:
        first_after = parts[1]

    print(f"  function signature/name: '{first_after}'")

    # Continue with empty function name to see what happens
    if first_after.endswith(':'):
        first_after = first_after[:-1]

    if '(' not in first_after:
        func_name = first_after
        sig = ((), None, None, {})
        print(f"  parsed as simple function: name='{func_name}'")
    else:
        print("  has parentheses, parsing as function with args")

    # Now check what tokens remain
    print(f"  remaining tokens: {tokens}")

    # The code will loop looking for content until enddef
    context = context + ('def',)
    content = []

    print("  Starting content loop...")
    while tokens:
        if isinstance(tokens[0], tuple) and tokens[0][0] == 'enddef':
            print(f"  Found enddef, returning def node")
            return ('def', start, func_name, sig, content), tokens[1:]

        print(f"  Processing token: {tokens[0]}")
        # This recursively calls parse_expr which might throw the error!
        next_chunk, tokens = tempita.parse_expr(tokens, name, context)
        content.append(next_chunk)

    print("  Reached end without enddef!")

tempita.parse_def = traced_parse_def

from Cython.Tempita import Template

print("Testing {{def }}{{enddef}}:")
try:
    template = Template("{{def }}{{enddef}}")
    print("Success!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")