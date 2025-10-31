import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-key',
        INSTALLED_APPS=[],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
    )

django.setup()

from django.db.models.sql.datastructures import Join
from django.db.models.sql.constants import INNER, LOUTER


class MockField:
    def get_joining_columns(self):
        return [("id", "fk_id")]

    def get_extra_restriction(self, table_alias, parent_alias):
        return None


field = MockField()

join_inner = Join(
    table_name="users",
    parent_alias="t1",
    table_alias="t2",
    join_type=INNER,
    join_field=field,
    nullable=False
)

join_outer = Join(
    table_name="users",
    parent_alias="t1",
    table_alias="t2",
    join_type=LOUTER,
    join_field=field,
    nullable=False
)

print(f"join_inner.join_type = {join_inner.join_type}")
print(f"join_outer.join_type = {join_outer.join_type}")
print(f"join_inner == join_outer: {join_inner == join_outer}")
print(f"hash(join_inner) == hash(join_outer): {hash(join_inner) == hash(join_outer)}")

join_set = {join_inner, join_outer}
print(f"len({{join_inner, join_outer}}): {len(join_set)}")

# Additional debugging
print(f"\njoin_inner.identity = {join_inner.identity}")
print(f"join_outer.identity = {join_outer.identity}")