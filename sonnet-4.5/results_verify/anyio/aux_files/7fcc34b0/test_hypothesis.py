import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
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
    print("Test completed")