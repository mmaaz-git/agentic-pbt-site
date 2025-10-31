import pandas as pd
import pandas.io.json as pj
import json

# Create a simple DataFrame
df = pd.DataFrame({'a': [1, 2, 3]})

# Test with primary_key=True
schema_true = pj.build_table_schema(df, index=True, primary_key=True)
print("With primary_key=True:")
print(json.dumps(schema_true, indent=2))
print(f"Type of primaryKey: {type(schema_true['primaryKey'])}")
print(f"Value of primaryKey: {schema_true['primaryKey']}")

print("\n" + "="*50 + "\n")

# Test with primary_key=False
schema_false = pj.build_table_schema(df, index=True, primary_key=False)
print("With primary_key=False:")
print(json.dumps(schema_false, indent=2))
print(f"Type of primaryKey: {type(schema_false['primaryKey'])}")
print(f"Value of primaryKey: {schema_false['primaryKey']}")

print("\n" + "="*50 + "\n")

# Test with primary_key=None (default)
schema_none = pj.build_table_schema(df, index=True, primary_key=None)
print("With primary_key=None (default):")
print(json.dumps(schema_none, indent=2))
print(f"Type of primaryKey: {type(schema_none['primaryKey'])}")
print(f"Value of primaryKey: {schema_none['primaryKey']}")