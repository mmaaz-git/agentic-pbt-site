import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

print("=" * 60)
print("Testing duplicate else clauses")
print("=" * 60)

content1 = "{{if x}}A{{else}}B{{else}}C{{endif}}"
print(f"Template 1: {content1}")

try:
    template1 = Template(content1)
    print("Template created successfully (should have raised error)")

    result_false = template1.substitute({'x': False})
    print(f"Result when x=False: {repr(result_false)}")

    result_true = template1.substitute({'x': True})
    print(f"Result when x=True: {repr(result_true)}")
except Exception as e:
    print(f"Error raised: {e}")

print("\n" + "=" * 60)
print("Testing elif after else")
print("=" * 60)

content2 = "{{if x}}A{{else}}B{{elif y}}C{{endif}}"
print(f"Template 2: {content2}")

try:
    template2 = Template(content2)
    print("Template created successfully (should have raised error)")

    result1 = template2.substitute({'x': False, 'y': True})
    print(f"Result when x=False, y=True: {repr(result1)}")

    result2 = template2.substitute({'x': False, 'y': False})
    print(f"Result when x=False, y=False: {repr(result2)}")

    result3 = template2.substitute({'x': True, 'y': True})
    print(f"Result when x=True, y=True: {repr(result3)}")

    result4 = template2.substitute({'x': True, 'y': False})
    print(f"Result when x=True, y=False: {repr(result4)}")
except Exception as e:
    print(f"Error raised: {e}")

print("\n" + "=" * 60)
print("Testing triple else clauses")
print("=" * 60)

content3 = "{{if x}}A{{else}}B{{else}}C{{else}}D{{endif}}"
print(f"Template 3: {content3}")

try:
    template3 = Template(content3)
    print("Template created successfully (should have raised error)")

    result_false = template3.substitute({'x': False})
    print(f"Result when x=False: {repr(result_false)}")

    result_true = template3.substitute({'x': True})
    print(f"Result when x=True: {repr(result_true)}")
except Exception as e:
    print(f"Error raised: {e}")