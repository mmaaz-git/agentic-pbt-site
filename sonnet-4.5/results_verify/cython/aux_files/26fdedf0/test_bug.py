import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import sysconfig

# First let's check what EXE normally returns
print(f"Normal EXE value: {sysconfig.get_config_var('EXE')!r}")

# Now mock it to return None
original_get_config_var = sysconfig.get_config_var

def mock_get_config_var(name):
    if name == 'EXE':
        return None
    return original_get_config_var(name)

sysconfig.get_config_var = mock_get_config_var

try:
    import importlib
    # Ensure we reload the module to use our mocked config
    if 'Cython.Build.BuildExecutable' in sys.modules:
        del sys.modules['Cython.Build.BuildExecutable']

    import Cython.Build.BuildExecutable
    from Cython.Build.BuildExecutable import EXE_EXT

    print(f"EXE_EXT value after mocking: {EXE_EXT!r}")

    # Try the concatenation that should fail
    basename = "test_program"
    try:
        exe_name = basename + EXE_EXT
        print(f"Concatenation succeeded: {exe_name!r}")
    except TypeError as e:
        print(f"BUG CONFIRMED: {e}")
        print("Cannot concatenate string with None")

finally:
    sysconfig.get_config_var = original_get_config_var