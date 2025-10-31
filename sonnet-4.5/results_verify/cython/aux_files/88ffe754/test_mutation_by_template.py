import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test if template code can mutate the dict
template = Template("{{py:x=100}}Modified: {{x}}", name="test.txt")
context = {'x': 42}

print("Before:", context)
result = template.substitute(context)
print("After:", context)
print("Result:", result)