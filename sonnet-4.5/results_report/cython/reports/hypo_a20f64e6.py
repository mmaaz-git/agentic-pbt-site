from hypothesis import given, strategies as st
from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options


@given(st.text(min_size=1), st.integers())
def test_backup_restore_round_trip_with_additions(key, value):
    original_backup = backup_Options()
    setattr(Options, key, value)
    restore_Options(original_backup)
    assert not hasattr(Options, key) or getattr(Options, key) == original_backup.get(key)

if __name__ == "__main__":
    test_backup_restore_round_trip_with_additions()