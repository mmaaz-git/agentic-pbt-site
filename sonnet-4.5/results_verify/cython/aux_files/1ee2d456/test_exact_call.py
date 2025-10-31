import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Tempita._tempita as tempita

# Backup original
original_parse_def = tempita.parse_def

def crash_parse_def(tokens, name, context):
    raise Exception("parse_def was called!")

# Replace it
tempita.parse_def = crash_parse_def

from Cython.Tempita import Template

print("Testing {{def }}{{enddef}}:")
try:
    template = Template("{{def }}{{enddef}}")
    print("No error!")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")