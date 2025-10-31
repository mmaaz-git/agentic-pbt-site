import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import Cython.Compiler.Tests.Utils as Utils
import Cython.Compiler.Options as Options


@given(st.text())
@settings(max_examples=10)  # Reduced for testing
def test_backup_restore_round_trip(s):
    original_backup = Utils.backup_Options()

    Options.test_attr = s

    modified_backup = Utils.backup_Options()
    Utils.restore_Options(modified_backup)

    restored_backup = Utils.backup_Options()
    assert modified_backup == restored_backup

    Utils.restore_Options(original_backup)

print("Running hypothesis test...")
try:
    test_backup_restore_round_trip()
    print("Test passed!")
except Exception as e:
    print(f"Test failed with: {type(e).__name__}: {e}")