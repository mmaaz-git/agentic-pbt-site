import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from hypothesis import given, strategies as st, settings
from fastapi.openapi.docs import get_swagger_ui_html

@given(st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.text(),
    min_size=1,
    max_size=5
))
@settings(max_examples=500)
def test_swagger_ui_parameters_no_script_tag_injection(params):
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="Test",
        swagger_ui_parameters=params
    )

    html_str = html.body.decode()

    for value in params.values():
        if "</script>" in str(value):
            assert "<\\/script>" in html_str or "&lt;/script&gt;" in html_str, \
                   f"</script> tag not properly escaped in: {value}"

# Run the test
if __name__ == "__main__":
    test_swagger_ui_parameters_no_script_tag_injection()
    print("Test completed")