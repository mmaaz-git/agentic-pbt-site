import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

# Import just the source module to test the actual Python code
from Cython.Tempita import _tempita

# Test the parse_def function directly
tokens = [('def ', (1, 0)), ('enddef', (1, 10))]
name = 'test'
context = ()

print("Testing parse_def with 'def ' (no function signature)")
print(f"tokens[0] = {repr(tokens[0])}")

try:
    first, start = tokens[0]
    print(f"first = {repr(first)}")
    parts = first.split(None, 1)
    print(f"first.split(None, 1) = {parts}")
    print(f"Length of parts: {len(parts)}")
    if len(parts) < 2:
        print("Would raise error: no function signature after 'def'")
    else:
        result = parts[1]
        print(f"parts[1] = {repr(result)}")
except IndexError as e:
    print(f"IndexError: {e}")