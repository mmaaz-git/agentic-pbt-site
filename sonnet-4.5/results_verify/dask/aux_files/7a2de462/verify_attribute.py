import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

# Test to verify the attribute name issue
test_val = 42
print(f"type(test_val) = {type(test_val)}")
print(f"dir(type(test_val)) showing __name__ related attributes:")
for attr in dir(type(test_val)):
    if 'name' in attr.lower():
        print(f"  - {attr}")

# Try accessing __name vs __name__
try:
    print(f"\ntype(test_val).__name = {type(test_val).__name}")
except AttributeError as e:
    print(f"\ntype(test_val).__name failed: {e}")

print(f"type(test_val).__name__ = {type(test_val).__name__}")