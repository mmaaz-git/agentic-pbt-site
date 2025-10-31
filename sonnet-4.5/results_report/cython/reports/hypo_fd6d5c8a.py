import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import sysconfig


@given(st.text(min_size=1, max_size=20))
def test_exe_ext_string_concatenation(basename):
    import importlib
    original_get_config_var = sysconfig.get_config_var

    def mock_get_config_var(name):
        if name == 'EXE':
            return None
        return original_get_config_var(name)

    sysconfig.get_config_var = mock_get_config_var

    try:
        if 'Cython.Build.BuildExecutable' in sys.modules:
            del sys.modules['Cython.Build.BuildExecutable']

        import Cython.Build.BuildExecutable

        from Cython.Build.BuildExecutable import EXE_EXT
        result = basename + EXE_EXT

    finally:
        sysconfig.get_config_var = original_get_config_var


if __name__ == '__main__':
    test_exe_ext_string_concatenation()