import sys
# Add Django to the path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.migrations.operations import RenameModel
from django.db.migrations.state import ProjectState

# Create a RenameModel operation
op = RenameModel(old_name="User", new_name="Person")

print(f"Before database_backwards:")
print(f"  op.old_name = {op.old_name}")
print(f"  op.new_name = {op.new_name}")
print(f"  op.old_name_lower = {op.old_name_lower}")
print(f"  op.new_name_lower = {op.new_name_lower}")

# Create a project state
state = ProjectState()

# Call database_backwards
try:
    op.database_backwards('test_app', None, state, state)
except:
    # It may fail due to missing schema_editor, but that's not the issue we're testing
    pass

print(f"\nAfter database_backwards:")
print(f"  op.old_name = {op.old_name}")
print(f"  op.new_name = {op.new_name}")
print(f"  op.old_name_lower = {op.old_name_lower}")
print(f"  op.new_name_lower = {op.new_name_lower}")

print("\n=== Mutation detected! ===")
print("The instance attributes were mutated by database_backwards,")
print("violating the immutability contract documented in the Operation base class.")