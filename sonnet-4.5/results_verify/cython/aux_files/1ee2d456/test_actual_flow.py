import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Insert a monkey patch to trace execution
import Cython.Tempita._tempita as tempita

original_parse_def = tempita.parse_def

def traced_parse_def(tokens, name, context):
    print(f"parse_def called with tokens: {tokens[:2] if len(tokens) >= 2 else tokens}")
    first, start = tokens[0]
    print(f"  first token: '{first}'")

    assert first.startswith('def ')

    # This is the problematic line
    try:
        parts = first.split(None, 1)
        print(f"  split result: {parts}")
        if len(parts) < 2:
            print(f"  ERROR: Would crash here with IndexError on parts[1]!")
            raise IndexError("list index out of range")
        first_after = parts[1]
        print(f"  first after split: '{first_after}'")
    except IndexError as e:
        print(f"  Caught IndexError: {e}")
        raise

    return original_parse_def(tokens, name, context)

tempita.parse_def = traced_parse_def

# Now test
from Cython.Tempita import Template

print("Testing {{def }}{{enddef}} with trace:")
print("="*50)

try:
    template = Template("{{def }}{{enddef}}")
    print("No error raised!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")