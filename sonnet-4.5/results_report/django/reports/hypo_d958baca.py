#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.template import TemplateSyntaxError
from django.templatetags.static import StaticNode


class MockToken:
    def __init__(self, contents):
        self.contents = contents

    def split_contents(self):
        return self.contents.split()


class MockParser:
    def compile_filter(self, token):
        return token


@given(st.lists(st.text(min_size=1, max_size=20), min_size=2, max_size=10))
@settings(max_examples=1000)
def test_static_node_handle_token_no_index_error(token_parts):
    token = MockToken(' '.join(token_parts))
    parser = MockParser()

    try:
        result = StaticNode.handle_token(parser, token)
    except (TemplateSyntaxError, AttributeError):
        pass
    except IndexError as e:
        raise AssertionError(f"IndexError with token parts: {token_parts}")


if __name__ == "__main__":
    test_static_node_handle_token_no_index_error()