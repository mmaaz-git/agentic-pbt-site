import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.management.templates import TemplateCommand

cmd = TemplateCommand()

# Test protocol-only URLs
test_urls = ["http://", "https://", "ftp://"]

for url in test_urls:
    print(f"\nTesting URL: {url!r}")
    print(f"is_url() returns: {cmd.is_url(url)}")

    # Show what filename would be extracted (from download method's cleanup_url logic)
    tmp = url.rstrip("/")
    filename = tmp.split("/")[-1]
    print(f"Extracted filename: {filename!r}")
    print(f"Filename contains colon: {':' in filename}")
    print(f"Filename length: {len(filename)}")

    # Check if this is valid on Windows
    if ':' in filename:
        print(f"ERROR: Filename {filename!r} is invalid on Windows (contains colon)")