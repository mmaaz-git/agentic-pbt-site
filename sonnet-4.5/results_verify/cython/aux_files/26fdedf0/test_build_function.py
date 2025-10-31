import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import sysconfig
import os

# Mock to return None
original_get_config_var = sysconfig.get_config_var

def mock_get_config_var(name):
    if name == 'EXE':
        return None
    return original_get_config_var(name)

sysconfig.get_config_var = mock_get_config_var

try:
    # Clear the module if it exists
    if 'Cython.Build.BuildExecutable' in sys.modules:
        del sys.modules['Cython.Build.BuildExecutable']

    import Cython.Build.BuildExecutable as BuildExecutable

    print(f"EXE_EXT value: {BuildExecutable.EXE_EXT!r}")

    # Try to use the build function with a dummy file
    # First let's create a minimal Cython file
    with open('/tmp/test_cython.pyx', 'w') as f:
        f.write('print("Hello from Cython")')

    # Now try to use build - this should fail on line 139
    try:
        result = BuildExecutable.build('/tmp/test_cython.pyx')
        print(f"Build succeeded: {result}")
    except TypeError as e:
        print(f"BUG IN BUILD FUNCTION: {e}")
        print(f"Error occurred when trying to create exe_file = basename + EXE_EXT")

    # Also test the clink function directly with a fake basename
    try:
        # This would fail on line 110 of clink function
        print("Testing clink function...")
        # We can't actually run clink without proper compilation, but we can check
        # that the string concatenation would fail
        basename = "test"
        exe_name = basename + BuildExecutable.EXE_EXT
    except TypeError as e:
        print(f"BUG IN STRING CONCATENATION: {e}")

finally:
    sysconfig.get_config_var = original_get_config_var
    # Clean up
    if os.path.exists('/tmp/test_cython.pyx'):
        os.remove('/tmp/test_cython.pyx')