import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.db.models.sql.where import WhereNode, XOR
from django.core.exceptions import FullResultSet


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


# Temporarily patch the WhereNode class to test the fix
original_as_sql = WhereNode.as_sql

def patched_as_sql(self, compiler, connection):
    # Apply the proposed fix: check for self.children before XOR emulation
    if self.connector == XOR and self.children and not connection.features.supports_logical_xor:
        # This is line 130 in the original with the fix applied
        import operator
        from functools import reduce
        from django.db.models.expressions import Case, When
        from django.db.models.functions import Mod
        from django.db.models.lookups import Exact

        lhs = self.__class__(self.children, 'OR')
        rhs_sum = reduce(
            operator.add,
            (Case(When(c, then=1), default=0) for c in self.children),
        )
        if len(self.children) > 2:
            rhs_sum = Mod(rhs_sum, 2)
        rhs = Exact(1, rhs_sum)
        return self.__class__([lhs, rhs], 'AND', self.negated).as_sql(
            compiler, connection
        )
    # Otherwise continue with normal flow
    return original_as_sql(self, compiler, connection)

# Apply the patch
WhereNode.as_sql = patched_as_sql

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
finally:
    # Restore original method
    WhereNode.as_sql = original_as_sql