#!/usr/bin/env python3
"""Test case to reproduce the bug report"""
import os
import tempfile
import pyximport

# First, reproduce the exact bug scenario
print("Testing the exact bug reproduction...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        with open(pyxbld_file, 'w') as f:
            f.write('''def make_setup_args():
    return {'extra_compile_args': ['-O3']}
''')

        ext, setup_args = pyximport.handle_special_build('test', pyxfile)
        print(f"Success! ext={ext}, setup_args={setup_args}")
except AttributeError as e:
    print(f"AttributeError occurred as expected: {e}")
except Exception as e:
    print(f"Different error occurred: {e}")

# Test the working case with both functions
print("\nTesting with both make_ext and make_setup_args...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test2.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test2.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        with open(pyxbld_file, 'w') as f:
            f.write('''from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])

def make_setup_args():
    return {'extra_compile_args': ['-O3']}
''')

        ext, setup_args = pyximport.handle_special_build('test2', pyxfile)
        print(f"Success! ext={ext}, setup_args={setup_args}")
except Exception as e:
    print(f"Error occurred: {e}")

# Test with only make_ext
print("\nTesting with only make_ext...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        pyxfile = os.path.join(tmpdir, 'test3.pyx')
        pyxbld_file = os.path.join(tmpdir, 'test3.pyxbld')

        with open(pyxfile, 'w') as f:
            f.write('# cython code')

        with open(pyxbld_file, 'w') as f:
            f.write('''from distutils.extension import Extension

def make_ext(modname, pyxfilename):
    return Extension(name=modname, sources=[pyxfilename])
''')

        ext, setup_args = pyximport.handle_special_build('test3', pyxfile)
        print(f"Success! ext={ext}, setup_args={setup_args}")
except Exception as e:
    print(f"Error occurred: {e}")