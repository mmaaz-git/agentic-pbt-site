import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from unittest.mock import Mock
from django.db.backends.base.operations import BaseDatabaseOperations

mock_conn = Mock()
mock_conn.ops.no_limit_value.return_value = 2**63 - 1
ops = BaseDatabaseOperations(connection=mock_conn)

sql = ops.limit_offset_sql(low_mark=10, high_mark=5)
print(f"Generated SQL: {sql}")

limit, offset = ops._get_limit_offset_params(low_mark=10, high_mark=5)
print(f"limit={limit}, offset={offset}")