import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-key',
        INSTALLED_APPS=[],
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
    )

django.setup()

from hypothesis import given, strategies as st
from django.db.models.sql.datastructures import Join
from django.db.models.sql.constants import INNER, LOUTER


class MockField:
    def get_joining_columns(self):
        return [("id", "fk_id")]

    def get_extra_restriction(self, table_alias, parent_alias):
        return None


@given(st.text(min_size=1, max_size=20))
def test_join_type_affects_equality(table_name):
    field = MockField()

    join_inner = Join(
        table_name=table_name,
        parent_alias="t1",
        table_alias="t2",
        join_type=INNER,
        join_field=field,
        nullable=False
    )

    join_outer = Join(
        table_name=table_name,
        parent_alias="t1",
        table_alias="t2",
        join_type=LOUTER,
        join_field=field,
        nullable=False
    )

    assert join_inner.join_type != join_outer.join_type
    assert join_inner != join_outer, "INNER JOIN should not equal LEFT OUTER JOIN"
    if join_inner == join_outer:
        assert hash(join_inner) != hash(join_outer), "Different joins should have different hashes"


if __name__ == "__main__":
    test_join_type_affects_equality()