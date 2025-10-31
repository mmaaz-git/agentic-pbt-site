import ast

# Read the actual source file
with open("/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py") as f:
    source = f.read()

tree = ast.parse(source)

for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
        print(f"Found function: {node.name}")
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                keys_repr = [ast.unparse(k) for k in child.keys]
                print(f"Total keys: {len(keys_repr)}")
                print(f"Unique keys: {len(set(keys_repr))}")
                print(f"Keys: {keys_repr}")

                # Check for duplicates
                seen = set()
                duplicates = []
                for key in keys_repr:
                    if key in seen:
                        duplicates.append(key)
                    seen.add(key)

                if duplicates:
                    print(f"DUPLICATE KEYS FOUND: {duplicates}")
                    assert len(keys_repr) == len(set(keys_repr)), \
                        f"Duplicate keys found: {duplicates}"