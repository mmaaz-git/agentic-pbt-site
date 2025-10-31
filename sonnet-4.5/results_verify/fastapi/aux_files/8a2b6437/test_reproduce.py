import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from fastapi.openapi.docs import get_swagger_ui_html

test_params = {"description": "</script><script>alert('XSS')</script><script>"}

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test API",
    swagger_ui_parameters=test_params
)

html_str = html.body.decode()
start = html_str.find("description")
end = html_str.find("presets", start)
print("Output between 'description' and 'presets':")
print(html_str[start:end])
print("\n---\n")
print("Expected output (from bug report):")
print('description": "</script><script>alert(\'XSS\')</script><script>",\n\n')