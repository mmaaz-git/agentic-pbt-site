import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.Cygdb import make_command_file

prefix_code = '\ud800'

try:
    result = make_command_file(None, prefix_code, no_import=True, skip_interpreter=False)
    print("No crash - unexpected")
except UnicodeEncodeError as e:
    print(f"Bug confirmed: {e}")