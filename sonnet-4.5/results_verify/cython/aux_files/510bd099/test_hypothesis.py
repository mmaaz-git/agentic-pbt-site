import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options


@given(st.integers(min_value=0, max_value=10))
@settings(max_examples=100)
def test_restore_restores_all_attributes(seed):
    backup = backup_Options()

    Options.new_test_attr = "test_value"
    Options.existing_attr_modified = True

    restore_Options(backup)

    assert not hasattr(Options, 'new_test_attr')

if __name__ == "__main__":
    test_restore_restores_all_attributes()