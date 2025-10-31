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
print("Result:")
print(result)
print("\n--- Analysis ---")
print("The template parsed successfully without raising an error.")
print("Only 'B' (first else clause) was included in the output.")
print("The second else clause 'C' was silently ignored.")