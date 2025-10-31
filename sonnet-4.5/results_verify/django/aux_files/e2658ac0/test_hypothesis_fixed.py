import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.postgresql', 'NAME': 'test'}},
        USE_TZ=True,
    )
    django.setup()

from django.db.backends.postgresql.operations import DatabaseOperations

class MockConnection:
    pass

ops = DatabaseOperations(MockConnection())

VALID_PG_EXTRACT_FIELDS = {
    'CENTURY', 'DAY', 'DECADE', 'DOW', 'DOY', 'EPOCH', 'HOUR',
    'ISODOW', 'ISOYEAR', 'JULIAN', 'MICROSECONDS', 'MILLENNIUM',
    'MILLISECONDS', 'MINUTE', 'MONTH', 'QUARTER', 'SECOND',
    'TIMEZONE', 'TIMEZONE_HOUR', 'TIMEZONE_MINUTE', 'WEEK', 'YEAR'
}

def test_field(lookup_type):
    """Test if the field is properly validated."""
    if lookup_type not in VALID_PG_EXTRACT_FIELDS:
        try:
            sql, params = ops.date_extract_sql(lookup_type, 'test_column', ())
            print(f"Bug found! Should reject invalid field: {lookup_type}, but got SQL: {sql}")
            return False  # Bug: invalid field was accepted
        except ValueError:
            return True  # Good: invalid field was rejected
    else:
        # Valid field, should work
        try:
            sql, params = ops.date_extract_sql(lookup_type, 'test_column', ())
            return True  # Good: valid field was accepted
        except ValueError:
            print(f"Bug: Valid field {lookup_type} was rejected!")
            return False

# Test specific failing inputs mentioned in bug report
print("Testing specific failing inputs mentioned in bug report:")
print("=" * 60)

test_inputs = ["WEEK_DAY", "ISO_WEEK_DAY", "ISO_YEAR", "INVALID_FIELD"]
for test_input in test_inputs:
    print(f"\nTesting: {test_input}")
    is_valid_pg_field = test_input in VALID_PG_EXTRACT_FIELDS
    print(f"  Is valid PostgreSQL field? {is_valid_pg_field}")

    result = test_field(test_input)
    if result:
        print(f"  ✓ Test passed")
    else:
        print(f"  ✗ Test failed")

print("\n\nTesting more invalid field patterns:")
print("=" * 60)

# Test more patterns that should fail but pass due to the regex
invalid_fields = [
    "WEEK_DAY",
    "ISO_WEEK_DAY",
    "ISO_YEAR",
    "INVALID_FIELD",
    "SOME_RANDOM_FIELD",
    "A_B_C_D",
    "TEST_EXTRACT_FIELD"
]

for field in invalid_fields:
    print(f"\nTesting invalid field: {field}")
    try:
        sql, params = ops.date_extract_sql(field, 'test_column', ())
        print(f"  Bug confirmed: Generated SQL: {sql}")
        print(f"  This would fail in PostgreSQL!")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")