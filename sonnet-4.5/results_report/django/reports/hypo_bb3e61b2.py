import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import pytest
from django.db.models.sql.where import WhereNode, XOR


class MockCompiler:
    def compile(self, child):
        return "1=1", []


class MockConnection:
    class features:
        supports_logical_xor = False


def test_empty_xor_node_no_native_support():
    """Test XOR with no children when database doesn't support XOR natively"""
    wn = WhereNode([], XOR)
    compiler = MockCompiler()
    conn = MockConnection()

    with pytest.raises(TypeError):
        wn.as_sql(compiler, conn)


if __name__ == "__main__":
    test_empty_xor_node_no_native_support()
    print("Test passed - TypeError was raised as expected")