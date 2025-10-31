import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template = Template("Value: {{x}}", name="test.txt")
context = {'x': 42}

print("Before:", list(context.keys()))
result = template.substitute(context)
print("After:", list(context.keys()))
print("Result:", result)