import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
        SECRET_KEY='test',
    )
    django.setup()

from hypothesis import given, settings as hypo_settings, strategies as st
from django.db.models import Index
from django.db.migrations.operations import AddIndex, RenameIndex

@given(
    model_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
    new_index_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("Lu", "Ll"))),
)
@hypo_settings(max_examples=1000)
def test_add_index_immutability_in_reduce(model_name, index_name, new_index_name):
    index = Index(fields=["id"], name=index_name)
    op = AddIndex(model_name=model_name, index=index)

    original_index_name = op.index.name

    rename_op = RenameIndex(model_name=model_name, old_name=index_name, new_name=new_index_name)
    result = op.reduce(rename_op, "app")

    assert op.index.name == original_index_name, f"Index name was mutated from {original_index_name} to {op.index.name}"

if __name__ == "__main__":
    # Run the test
    test_add_index_immutability_in_reduce()