import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.models.sql.where import WhereNode, XOR


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

# Try to generate SQL - this should crash
try:
    result = wn.as_sql(compiler, conn)
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()