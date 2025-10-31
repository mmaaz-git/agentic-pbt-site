import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'

from django.template.backends.jinja2 import get_exception_info


class MockException:
    def __init__(self, lineno, source, filename, message):
        self.lineno = lineno
        self.source = source
        self.filename = filename
        self.message = message


exc = MockException(
    lineno=10,
    source="line 1\nline 2\nline 3",
    filename="test.html",
    message="test error"
)

info = get_exception_info(exc)
print("Success! Info:", info)