import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# First patch the WhereNode before importing to test fix
import django.db.models.sql.where as where_module

# Read the original code
with open('/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/db/models/sql/where.py', 'r') as f:
    lines = f.readlines()

# Apply the fix at line 130 (index 129)
# Change: if self.connector == XOR and not connection.features.supports_logical_xor:
# To:     if self.connector == XOR and self.children and not connection.features.supports_logical_xor:
lines[129] = '        if self.connector == XOR and self.children and not connection.features.supports_logical_xor:\n'

# Write patched version
patched_code = ''.join(lines)

# Execute patched code
exec(patched_code, where_module.__dict__)

# Now test with the patched version
from django.db.models.sql.where import WhereNode, XOR
from django.core.exceptions import FullResultSet


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


# Create an empty WhereNode with XOR connector
wn = WhereNode([], XOR)
compiler = MockCompiler()
conn = MockConnection()

# Try to generate SQL - with the fix, should raise FullResultSet
try:
    result = wn.as_sql(compiler, conn)
    print(f"Result: {result}")
except FullResultSet:
    print("SUCCESS: FullResultSet exception raised as expected with the fix applied")
except Exception as e:
    print(f"FAILED: Unexpected exception - {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()