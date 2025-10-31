import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import tempfile
import os
from Cython.Tempita import Template

with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
    f.write("Hello World")
    filename = f.name

try:
    template = Template.from_filename(filename)
    print("Template created successfully")
except TypeError as e:
    print(f"TypeError: {e}")
finally:
    os.unlink(filename)