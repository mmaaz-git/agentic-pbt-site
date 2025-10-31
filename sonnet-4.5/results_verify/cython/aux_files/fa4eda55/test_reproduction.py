#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import sub

# Test case from bug report
content = "Template name: {{__name}}"
result = sub(content, __name='mytemplate.html')

print(f"Result: '{result}'")
print(f"Expected: 'Template name: ' (empty)")
print(f"Actual: 'Template name: mytemplate.html'")
print()

# Additional test without __name variable
content2 = "No name here"
result2 = sub(content2, __name='test.html')
print(f"Test without __name in template: '{result2}'")
print()

# Test to confirm __name is indeed accessible as variable
content3 = "Name is: {{__name}} and foo is: {{foo}}"
result3 = sub(content3, __name='template.html', foo='bar')
print(f"Multiple variables test: '{result3}'")
print()

if result == "Template name: mytemplate.html":
    print("BUG CONFIRMED: __name parameter is accessible as template variable")
else:
    print("BUG NOT REPRODUCED: __name is not accessible")