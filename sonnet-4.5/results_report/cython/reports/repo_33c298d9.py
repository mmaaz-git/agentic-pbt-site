import sys
import os
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import handle_special_build

with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "module.pyx")
    pyxbld = os.path.join(tmpdir, "module.pyxbld")

    # Create empty .pyx file
    open(pyxfile, 'w').close()

    # Create .pyxbld file with only make_setup_args (no make_ext)
    with open(pyxbld, 'w') as f:
        f.write("""
def make_setup_args():
    return {'script_args': ['--verbose']}
""")

    # This should crash with AttributeError
    ext, setup_args = handle_special_build("module", pyxfile)