import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import io
from Cython.Tempita._tempita import fill_command

old_stdin = sys.stdin
old_stdout = sys.stdout

try:
    sys.stdin = io.StringIO("{{x}}")
    sys.stdout = io.StringIO()

    args = ['-', 'py:x=42']
    fill_command(args)

    result = sys.stdout.getvalue()
    print(f"Result: {result!r}")
finally:
    sys.stdin = old_stdin
    sys.stdout = old_stdout