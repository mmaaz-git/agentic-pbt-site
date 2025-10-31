import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env')

import os
import tempfile
import pyximport
from pathlib import Path

with tempfile.NamedTemporaryFile(mode='w', suffix='.pyx', delete=False) as f:
    f.write("# dummy\n")
    path = f.name

try:
    print("Testing bytes path:")
    pyximport.get_distutils_extension('mymod', path.encode('utf-8'))
except TypeError as e:
    print(f"  Bytes bug: {e}")

try:
    print("\nTesting Path object:")
    pyximport.get_distutils_extension('mymod', Path(path))
except AttributeError as e:
    print(f"  Path bug: {e}")
finally:
    os.unlink(path)