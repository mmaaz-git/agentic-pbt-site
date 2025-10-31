import os
import tempfile
import sys

# Add the pyximport to path if needed
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
import pyximport

with tempfile.TemporaryDirectory() as tmpdir:
    pyx_file = os.path.join(tmpdir, 'example.pyx')
    pyxbld_file = os.path.join(tmpdir, 'example.pyxbld')

    with open(pyx_file, 'w') as f:
        f.write('def hello(): return "world"')

    with open(pyxbld_file, 'w') as f:
        f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

    ext, setup_args = pyximport.handle_special_build('example', pyx_file)
    print(f"Extension: {ext}")
    print(f"Setup args: {setup_args}")