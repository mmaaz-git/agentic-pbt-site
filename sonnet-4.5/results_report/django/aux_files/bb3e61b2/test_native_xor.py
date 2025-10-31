import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.models.sql.where import WhereNode, XOR
from django.core.exceptions import FullResultSet


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnectionWithXOR:
    class features:
        supports_logical_xor = True  # Database supports native XOR


# Create an empty WhereNode with XOR connector
wn = WhereNode([], XOR)
compiler = MockCompiler()
conn = MockConnectionWithXOR()

# Try to generate SQL - should raise FullResultSet for empty WHERE
try:
    result = wn.as_sql(compiler, conn)
    print(f"Result: {result}")
except FullResultSet:
    print("FullResultSet exception raised as expected for empty WHERE with native XOR support")
except Exception as e:
    print(f"Unexpected exception - {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()