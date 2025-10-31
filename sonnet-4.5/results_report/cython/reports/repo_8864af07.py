import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import tempfile
import os
from Cython.Tempita import Template

# Create a temporary file with simple content
with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
    f.write("Hello World")
    filename = f.name

try:
    # Try to create a template without specifying encoding
    template = Template.from_filename(filename)
    result = template.substitute({})
    print(f"Success: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
finally:
    os.unlink(filename)