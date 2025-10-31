import sys
import os

# Remove the .so file temporarily
so_file = '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.cpython-313-x86_64-linux-gnu.so'
so_backup = so_file + '.bak'

if os.path.exists(so_file):
    os.rename(so_file, so_backup)
    print(f"Renamed {so_file} to {so_backup}")

# Now import and test
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template
from Cython.Tempita._tempita import TemplateError

print("\nTesting {{def }} with Python source (not compiled):")
content = "{{def }}{{enddef}}"
try:
    template = Template(content)
    print("   No error raised - unexpected!")
except IndexError as e:
    print(f"   IndexError raised: {e}")
except TemplateError as e:
    print(f"   TemplateError raised: {e}")
except Exception as e:
    print(f"   {type(e).__name__}: {e}")

# Restore the .so file
if os.path.exists(so_backup):
    os.rename(so_backup, so_file)
    print(f"\nRestored {so_file}")