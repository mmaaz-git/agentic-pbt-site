import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import sysconfig

# Store original function
original_get_config_var = sysconfig.get_config_var

def mock_get_config_var(name):
    if name == 'EXE':
        return None
    return original_get_config_var(name)

# Apply mock
sysconfig.get_config_var = mock_get_config_var

try:
    import importlib
    # Force reimport to use our mocked function
    if 'Cython.Build.BuildExecutable' in sys.modules:
        del sys.modules['Cython.Build.BuildExecutable']

    import Cython.Build.BuildExecutable
    from Cython.Build.BuildExecutable import EXE_EXT, build

    print(f"EXE_EXT value: {EXE_EXT!r}")
    print(f"Type of EXE_EXT: {type(EXE_EXT)}")

    # Try to use it as the module does
    basename = "test_program"
    print(f"\nAttempting string concatenation: basename + EXE_EXT")
    print(f"  basename = {basename!r}")
    print(f"  EXE_EXT = {EXE_EXT!r}")

    # This will fail with TypeError
    exe_name = basename + EXE_EXT
    print(f"Result: {exe_name!r}")

except TypeError as e:
    print(f"\nBUG CONFIRMED - TypeError occurred:")
    print(f"  Error: {e}")
    print(f"  Cannot concatenate string '{basename}' with None")

finally:
    # Restore original function
    sysconfig.get_config_var = original_get_config_var