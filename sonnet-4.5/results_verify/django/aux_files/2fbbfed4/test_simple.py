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

index = Index(fields=["id"], name="old_index")
add_op = AddIndex(model_name="MyModel", index=index)

print(f"Before reduce: {add_op.index.name}")

rename_op = RenameIndex(model_name="MyModel", old_name="old_index", new_name="new_index")
add_op.reduce(rename_op, "app")

print(f"After reduce: {add_op.index.name}")