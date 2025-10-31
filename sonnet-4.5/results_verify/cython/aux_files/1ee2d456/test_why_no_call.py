import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Tempita._tempita as tempita

original_parse_expr = tempita.parse_expr

def traced_parse_expr(tokens, name, context=()):
    if len(tokens) > 0 and isinstance(tokens[0], tuple):
        expr, pos = tokens[0]
        print(f"parse_expr: expr='{expr}'")

        # Check the condition
        if expr.startswith('def '):
            print(f"  -> Should call parse_def (expr starts with 'def ')")
        elif expr == 'def':
            print(f"  -> expr is exactly 'def', doesn't start with 'def '")
        elif expr in ('endif', 'endfor', 'enddef'):
            print(f"  -> Will raise TemplateError for unexpected {expr}")

    return original_parse_expr(tokens, name, context)

tempita.parse_expr = traced_parse_expr

from Cython.Tempita import Template

print("Testing {{def }}{{enddef}}:")
try:
    template = Template("{{def }}{{enddef}}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting {{def}}{{enddef}}:")
try:
    template = Template("{{def}}{{enddef}}")
except Exception as e:
    print(f"Error: {e}")