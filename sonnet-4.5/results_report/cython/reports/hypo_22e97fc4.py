import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings, Phase
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options, check_global_options
from Cython.Compiler import Options

@given(attr_name=st.text(min_size=1, max_size=50).filter(lambda x: x.isidentifier() and not x.startswith('_')),
       attr_value=st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans(), st.none()))
@settings(phases=(Phase.generate, Phase.target, Phase.shrink), max_examples=10)
def test_backup_restore_options_roundtrip(attr_name, attr_value):
    """Test that backup_Options and restore_Options properly handle new attributes."""
    # Create a backup of current state
    backup = backup_Options()

    # Add a new attribute
    setattr(Options, attr_name, attr_value)

    # Verify the attribute exists
    assert hasattr(Options, attr_name)
    assert getattr(Options, attr_name) == attr_value

    # Restore to original state
    restore_Options(backup)

    # Verify the new attribute was removed
    assert not hasattr(Options, attr_name), f"New attribute '{attr_name}' should have been removed"

    # Verify all original options are restored
    assert check_global_options(backup) == "", "Original options should be restored"

if __name__ == "__main__":
    test_backup_restore_options_roundtrip()