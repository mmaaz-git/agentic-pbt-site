"""Test to reproduce the pyximport.handle_special_build bug"""
import os
import tempfile
import sys
import traceback

# Add the cython env to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import pyximport

def test_handle_special_build_with_only_setup_args():
    """Test case from the bug report - pyxbld file with only make_setup_args"""
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_file = os.path.join(tmpdir, 'example.pyx')
        pyxbld_file = os.path.join(tmpdir, 'example.pyxbld')

        # Create the .pyx file
        with open(pyx_file, 'w') as f:
            f.write('def hello(): return "world"')

        # Create the .pyxbld file with ONLY make_setup_args (no make_ext)
        with open(pyxbld_file, 'w') as f:
            f.write('def make_setup_args():\n    return {"script_args": ["--verbose"]}')

        print(f"Created files:")
        print(f"  {pyx_file}")
        print(f"  {pyxbld_file}")

        try:
            # This should work according to the documentation, but will crash
            ext, setup_args = pyximport.handle_special_build('example', pyx_file)
            print(f"Success! ext={ext}, setup_args={setup_args}")
            return True
        except AttributeError as e:
            print(f"AttributeError as expected: {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Unexpected error: {e}")
            traceback.print_exc()
            return False

def test_handle_special_build_with_both():
    """Test case with both make_ext and make_setup_args - should work"""
    with tempfile.TemporaryDirectory() as tmpdir:
        pyx_file = os.path.join(tmpdir, 'example2.pyx')
        pyxbld_file = os.path.join(tmpdir, 'example2.pyxbld')

        # Create the .pyx file
        with open(pyx_file, 'w') as f:
            f.write('def hello(): return "world"')

        # Create the .pyxbld file with BOTH make_ext and make_setup_args
        with open(pyxbld_file, 'w') as f:
            f.write('''from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {"script_args": ["--verbose"]}
''')

        print(f"\nCreated files:")
        print(f"  {pyx_file}")
        print(f"  {pyxbld_file}")

        try:
            ext, setup_args = pyximport.handle_special_build('example2', pyx_file)
            print(f"Success! ext={ext}, setup_args={setup_args}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: pyxbld with only make_setup_args (should fail with current code)")
    print("=" * 60)
    result1 = test_handle_special_build_with_only_setup_args()

    print("\n" + "=" * 60)
    print("Test 2: pyxbld with both make_ext and make_setup_args (should work)")
    print("=" * 60)
    result2 = test_handle_special_build_with_both()

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Test 1 (only make_setup_args): {'PASSED' if result1 else 'FAILED (as expected)'}")
    print(f"  Test 2 (both functions): {'PASSED' if result2 else 'FAILED'}")