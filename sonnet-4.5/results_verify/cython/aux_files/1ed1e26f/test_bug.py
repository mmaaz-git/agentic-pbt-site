import sys
import os
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import handle_special_build

# Test the bug: pyxbld with only make_setup_args
with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "module.pyx")
    pyxbld = os.path.join(tmpdir, "module.pyxbld")

    open(pyxfile, 'w').close()

    with open(pyxbld, 'w') as f:
        f.write("""
def make_setup_args():
    return {'script_args': ['--verbose']}
""")

    try:
        ext, setup_args = handle_special_build("module", pyxfile)
        print("SUCCESS: No error occurred")
        print(f"ext = {ext}")
        print(f"setup_args = {setup_args}")
    except AttributeError as e:
        print(f"ATTRIBUTEERROR: {e}")
    except Exception as e:
        print(f"OTHER ERROR: {type(e).__name__}: {e}")