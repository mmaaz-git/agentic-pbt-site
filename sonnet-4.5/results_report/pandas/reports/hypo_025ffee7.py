from hypothesis import given, strategies as st
import ast


def test_no_duplicate_keys_in_dict_literal():
    pandas_file = "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py"

    with open(pandas_file) as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
            for child in ast.walk(node):
                if isinstance(child, ast.Dict):
                    keys_repr = [ast.unparse(k) for k in child.keys]
                    assert len(keys_repr) == len(set(keys_repr)), \
                        f"Duplicate keys found: {keys_repr}"


if __name__ == "__main__":
    test_no_duplicate_keys_in_dict_literal()