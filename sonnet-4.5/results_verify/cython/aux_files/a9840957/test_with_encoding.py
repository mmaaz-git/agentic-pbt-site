import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import tempfile
import os
from Cython.Tempita import Template

with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
    f.write("Hello World")
    filename = f.name

try:
    # Test with explicit encoding
    template = Template.from_filename(filename, encoding='utf-8')
    result = template.substitute({})
    print(f"With encoding='utf-8': Success! Result: '{result}'")
except Exception as e:
    print(f"With encoding='utf-8': Failed with {type(e).__name__}: {e}")
finally:
    os.unlink(filename)