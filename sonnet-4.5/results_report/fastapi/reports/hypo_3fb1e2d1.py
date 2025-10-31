#!/usr/bin/env python3
"""
Property-based test for FastAPI openapi.docs XSS vulnerability
"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html

@settings(max_examples=500)
@given(st.text(min_size=1, max_size=100), st.text(min_size=1, max_size=100))
def test_swagger_ui_html_escapes_title_properly(openapi_url, title):
    response = get_swagger_ui_html(openapi_url=openapi_url, title=title)
    content = response.body.decode('utf-8')

    title_tag_start = content.find('<title>')
    title_tag_end = content.find('</title>')

    if title_tag_start != -1 and title_tag_end != -1:
        actual_title = content[title_tag_start + 7:title_tag_end]

        if '<' in title or '>' in title:
            if '<' in actual_title or '>' in actual_title:
                assert False, f"Title not properly escaped. Got {actual_title!r} from input {title!r}"

if __name__ == "__main__":
    test_swagger_ui_html_escapes_title_properly()