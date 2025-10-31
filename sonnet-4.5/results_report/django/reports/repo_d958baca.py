#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.templatetags.static import StaticNode


class MockToken:
    def __init__(self, contents):
        self.contents = contents

    def split_contents(self):
        return self.contents.split()


class MockParser:
    def compile_filter(self, token):
        return token


# Create test case that triggers the IndexError
parser = MockParser()
token = MockToken('x as y')

print("Testing StaticNode.handle_token with token: 'x as y'")
print(f"Token split_contents() returns: {token.split_contents()}")
print()

try:
    result = StaticNode.handle_token(parser, token)
    print(f"Success: returned {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()