import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test with bytes content
bytes_content = b"Hello {{name}}"
template = Template(bytes_content)
result = template.substitute({'name': 'World'})
print(result)