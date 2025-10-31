import sys
import os
import tempfile
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from pyximport.pyximport import handle_special_build

print("Testing all combinations of make_ext and make_setup_args:")
print("=" * 60)

# Test 1: pyxbld with only make_ext
print("\n1. ONLY make_ext:")
with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "test1.pyx")
    pyxbld = os.path.join(tmpdir, "test1.pyxbld")

    open(pyxfile, 'w').close()

    with open(pyxbld, 'w') as f:
        f.write("""
from distutils.extension import Extension
def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])
""")

    try:
        ext, setup_args = handle_special_build("test1", pyxfile)
        print(f"   SUCCESS: ext={ext}, setup_args={setup_args}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

# Test 2: pyxbld with only make_setup_args
print("\n2. ONLY make_setup_args:")
with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "test2.pyx")
    pyxbld = os.path.join(tmpdir, "test2.pyxbld")

    open(pyxfile, 'w').close()

    with open(pyxbld, 'w') as f:
        f.write("""
def make_setup_args():
    return {'script_args': ['--verbose']}
""")

    try:
        ext, setup_args = handle_special_build("test2", pyxfile)
        print(f"   SUCCESS: ext={ext}, setup_args={setup_args}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

# Test 3: pyxbld with both make_ext and make_setup_args
print("\n3. BOTH make_ext AND make_setup_args:")
with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "test3.pyx")
    pyxbld = os.path.join(tmpdir, "test3.pyxbld")

    open(pyxfile, 'w').close()

    with open(pyxbld, 'w') as f:
        f.write("""
from distutils.extension import Extension
def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {'script_args': ['--verbose']}
""")

    try:
        ext, setup_args = handle_special_build("test3", pyxfile)
        print(f"   SUCCESS: ext={ext}, setup_args={setup_args}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")

# Test 4: No pyxbld file
print("\n4. NO pyxbld file:")
with tempfile.TemporaryDirectory() as tmpdir:
    pyxfile = os.path.join(tmpdir, "test4.pyx")
    # No pyxbld file created

    open(pyxfile, 'w').close()

    try:
        ext, setup_args = handle_special_build("test4", pyxfile)
        print(f"   SUCCESS: ext={ext}, setup_args={setup_args}")
    except Exception as e:
        print(f"   ERROR: {type(e).__name__}: {e}")