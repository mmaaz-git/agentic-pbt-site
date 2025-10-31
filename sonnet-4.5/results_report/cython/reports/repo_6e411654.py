import sys
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Debugger.DebugWriter import is_valid_tag, CythonDebugWriter

tag = "0"
print(f"is_valid_tag('{tag}') = {is_valid_tag(tag)}")

with tempfile.TemporaryDirectory() as tmpdir:
    writer = CythonDebugWriter(tmpdir)
    writer.start(tag)