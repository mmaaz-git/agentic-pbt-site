#!/usr/bin/env python3
"""Understanding the exact code flow"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

# Let's trace through the code manually for "10."

var = "10."

# Line 822: if "." in var or "e" in var.lower():
print(f'"." in "{var}": {"." in var}')  # True

# Line 823: self.literal = float(var)
literal = float(var)
print(f'float("{var}") = {literal}')  # 10.0

# Line 825: if var[-1] == ".":
print(f'"{var}"[-1] == ".": {var[-1] == "."}')  # True

# Line 826: raise ValueError
# This SHOULD raise ValueError here, but let's trace what would happen if it didn't
print("\nComment in the code says '# \"2.\" is invalid'")
print("And line 826 should raise ValueError when var ends with '.'")
print("But there's a problem...")

# Let's see what ACTUALLY happens in the code
from django.template import Variable

print("\n" + "="*60)
print("What actually happens in Django Variable:")
print("="*60)

# Looking at line 822-828, it seems like the flow is:
# 1. If "." in var or "e" in var.lower():
# 2.     self.literal = float(var)  # This succeeds for "10."
# 3.     if var[-1] == ".":
# 4.         raise ValueError  # This SHOULD happen
# 5. else:
# 6.     self.literal = int(var)

# But then at line 829 we have:
# except ValueError:
#     ... code continues to set lookups ...

# So when ValueError is raised on line 826, it's CAUGHT by line 829!
# This means the literal=10.0 assignment from line 823 persists,
# but then execution continues at line 830 which eventually sets lookups

print("The issue is that:")
print("1. Line 823: self.literal = float('10.') sets literal to 10.0")
print("2. Line 826: raise ValueError is supposed to reject trailing period")
print("3. Line 829: except ValueError catches this error")
print("4. But self.literal is ALREADY set to 10.0 from line 823!")
print("5. Execution continues at line 830 and eventually sets self.lookups at line 848")
print("\nSo both literal AND lookups end up being set!")