import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case 1: Duplicate else clauses
print("Test 1: Duplicate else clauses")
print("-" * 40)
content1 = "{{if x}}A{{else}}B{{else}}C{{endif}}"
try:
    template1 = Template(content1)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False: {template1.substitute({'x': False})}")
    print(f"When x=True: {template1.substitute({'x': True})}")
except Exception as e:
    print(f"Error (expected): {e}")

print("\nTest 2: elif after else")
print("-" * 40)
content2 = "{{if x}}A{{else}}B{{elif y}}C{{endif}}"
try:
    template2 = Template(content2)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False, y=True: {template2.substitute({'x': False, 'y': True})}")
    print(f"When x=False, y=False: {template2.substitute({'x': False, 'y': False})}")
    print(f"When x=True, y=True: {template2.substitute({'x': True, 'y': True})}")
except Exception as e:
    print(f"Error (expected): {e}")

print("\nTest 3: Multiple duplicate else clauses")
print("-" * 40)
content3 = "{{if x}}A{{else}}B{{else}}C{{else}}D{{else}}E{{endif}}"
try:
    template3 = Template(content3)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False: {template3.substitute({'x': False})}")
    print(f"When x=True: {template3.substitute({'x': True})}")
except Exception as e:
    print(f"Error (expected): {e}")