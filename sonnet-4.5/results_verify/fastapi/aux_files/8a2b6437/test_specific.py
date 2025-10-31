import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from fastapi.openapi.docs import get_swagger_ui_html

# Test the specific failing input
test_params = {"description": "</script><script>alert('XSS')</script><script>"}

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test",
    swagger_ui_parameters=test_params
)

html_str = html.body.decode()

# Check if the </script> tag is properly escaped
if "</script>" in str(test_params["description"]):
    if "<\\/script>" in html_str or "&lt;/script&gt;" in html_str:
        print("PASS: </script> tag properly escaped")
    else:
        print("FAIL: </script> tag not properly escaped")
        print("\nSearching for the description value in HTML:")
        start = html_str.find("description")
        if start != -1:
            end = start + 200  # Show some context
            print(f"Found at position {start}:")
            print(html_str[start:end])