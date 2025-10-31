import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        SECRET_KEY='test',
    )
    django.setup()

from django.db.models import Index
from django.db.migrations.operations import AddIndex, RenameIndex

# Create an AddIndex operation with an index
index = Index(fields=["id"], name="old_index")
add_op = AddIndex(model_name="MyModel", index=index)

# Store original index name for verification
original_name = add_op.index.name
print(f"Before reduce: {add_op.index.name}")

# Create a RenameIndex operation
rename_op = RenameIndex(model_name="MyModel", old_name="old_index", new_name="new_index")

# Call reduce, which should NOT mutate the original operation
result = add_op.reduce(rename_op, "app")

print(f"After reduce: {add_op.index.name}")

# Verify the mutation
if add_op.index.name != original_name:
    print(f"ERROR: Index name was mutated from {original_name} to {add_op.index.name}")
    print("This violates the immutability contract of migration operations.")
else:
    print("SUCCESS: Index name remained unchanged.")