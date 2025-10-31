import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Configure Django settings first to avoid circular imports
import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    DATABASES={
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': ':memory:',
        }
    },
    INSTALLED_APPS=[],
    USE_TZ=False,
)
django.setup()

from unittest.mock import Mock
from hypothesis import given, strategies as st, settings, example
from django.db.backends.base.operations import BaseDatabaseOperations


@given(
    st.integers(min_value=0, max_value=10000),
    st.integers(min_value=0, max_value=10000)
)
@example(low_mark=10, high_mark=5)
@settings(max_examples=1000)
def test_limit_offset_sql_no_negative_limit(low_mark, high_mark):
    mock_conn = Mock()
    mock_conn.ops.no_limit_value.return_value = 2**63 - 1
    ops = BaseDatabaseOperations(connection=mock_conn)

    limit, offset = ops._get_limit_offset_params(low_mark, high_mark)

    assert limit is None or limit >= 0, \
        f"Negative limit: low_mark={low_mark}, high_mark={high_mark}, limit={limit}"


if __name__ == "__main__":
    test_limit_offset_sql_no_negative_limit()