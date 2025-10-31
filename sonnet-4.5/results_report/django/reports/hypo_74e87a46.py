import sys
# Add Django to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from django.db.migrations.operations import RenameModel

@given(
    old_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
    new_name=st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))),
)
def test_renamemodel_immutability(old_name, new_name):
    assume(old_name != new_name)

    op = RenameModel(old_name=old_name, new_name=new_name)

    # Store the original values
    original_old_name = op.old_name
    original_new_name = op.new_name
    original_old_name_lower = op.old_name_lower
    original_new_name_lower = op.new_name_lower

    from django.db.migrations.state import ProjectState
    state = ProjectState()

    # Call database_backwards - it might fail due to missing schema_editor,
    # but we're testing whether it mutates the instance
    try:
        op.database_backwards('test_app', None, state, state)
    except:
        pass

    # Check that the instance attributes were not mutated
    assert op.old_name == original_old_name, f"old_name was mutated from {original_old_name} to {op.old_name}"
    assert op.new_name == original_new_name, f"new_name was mutated from {original_new_name} to {op.new_name}"
    assert op.old_name_lower == original_old_name_lower, f"old_name_lower was mutated"
    assert op.new_name_lower == original_new_name_lower, f"new_name_lower was mutated"

if __name__ == "__main__":
    # Run the test
    test_renamemodel_immutability()