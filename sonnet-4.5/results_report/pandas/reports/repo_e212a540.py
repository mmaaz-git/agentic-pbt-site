import ast
import sys

# Read the actual pandas source file to analyze it
pandas_file = "/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/_util.py"

with open(pandas_file, 'r') as f:
    source = f.read()

# Parse the source code
tree = ast.parse(source)

# Find the _arrow_dtype_mapping function
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_arrow_dtype_mapping":
        print(f"Found function: {node.name}")
        # Look for dictionary literals in the function
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                # Extract the keys
                keys = []
                for k in child.keys:
                    if isinstance(k, ast.Call):
                        # Get the function call representation
                        keys.append(ast.unparse(k))

                print(f"\nTotal keys in dictionary: {len(keys)}")
                print(f"Unique keys: {len(set(keys))}")

                # Check for duplicates
                seen = set()
                duplicates = []
                for key in keys:
                    if key in seen:
                        duplicates.append(key)
                    else:
                        seen.add(key)

                if duplicates:
                    print(f"\nDuplicate keys found: {duplicates}")
                    print("\nAll keys in order:")
                    for i, key in enumerate(keys, 1):
                        print(f"  {i:2}. {key}")
                else:
                    print("No duplicate keys found")