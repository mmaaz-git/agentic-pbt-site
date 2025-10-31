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

# Now we can import the operations module
from django.db.backends.base.operations import BaseDatabaseOperations
from unittest.mock import Mock

# Create a mock connection to avoid needing a real database
mock_conn = Mock()
mock_conn.ops.no_limit_value.return_value = 2**63 - 1
ops = BaseDatabaseOperations(connection=mock_conn)

# Call the public API with high_mark < low_mark
low_mark = 10
high_mark = 5

print(f"Input: low_mark={low_mark}, high_mark={high_mark}")
print()

# Generate SQL using the public limit_offset_sql method
sql = ops.limit_offset_sql(low_mark=low_mark, high_mark=high_mark)
print(f"Generated SQL clause: '{sql}'")
print()

# Also show the internal calculation
limit, offset = ops._get_limit_offset_params(low_mark=low_mark, high_mark=high_mark)
print(f"Internal calculation:")
print(f"  limit={limit} (computed as high_mark - offset = {high_mark} - {low_mark} = {limit})")
print(f"  offset={offset}")
print()

print(f"Issue: LIMIT {limit} is negative, which is invalid SQL.")
print("Most databases will reject this query with a syntax error.")