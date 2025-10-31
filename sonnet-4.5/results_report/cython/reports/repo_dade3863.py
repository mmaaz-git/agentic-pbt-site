import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = """
{{if False}}
  A
{{else}}
  B
{{else}}
  C
{{endif}}
"""

template = Template(content)
result = template.substitute({})
print(result)