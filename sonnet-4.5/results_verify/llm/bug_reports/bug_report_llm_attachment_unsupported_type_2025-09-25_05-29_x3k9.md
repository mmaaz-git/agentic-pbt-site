# Bug Report: llm.default_plugins._attachment Unsupported MIME Types Treated as Audio

**Target**: `llm.default_plugins.openai_models._attachment`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_attachment` function incorrectly treats unsupported MIME types (e.g., `text/plain`, `video/mp4`) as audio attachments due to an unconditional else block that assumes all non-image types are audio.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import _attachment
from unittest.mock import Mock


@given(st.sampled_from(["text/plain", "text/html", "application/json", "video/mp4"]))
def test_attachment_should_handle_unsupported_types(mime_type):
    attachment = Mock()
    attachment.url = None
    attachment.resolve_type = Mock(return_value=mime_type)
    attachment.base64_content = Mock(return_value="dGVzdA==")

    result = _attachment(attachment)

    assert result["type"] != "input_audio", \
        f"{mime_type} should not be treated as audio"
```

**Failing input**: `mime_type="text/plain"` (and any other unsupported type)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import _attachment
from unittest.mock import Mock

attachment = Mock()
attachment.url = None
attachment.resolve_type = Mock(return_value="text/plain")
attachment.base64_content = Mock(return_value="SGVsbG8gV29ybGQ=")

result = _attachment(attachment)

print(f"Result: {result}")

assert result == {
    'type': 'input_audio',
    'input_audio': {'data': 'SGVsbG8gV29ybGQ=', 'format': 'mp3'}
}
```

## Why This Is A Bug

The function uses an unconditional else block (line 474) that assumes any attachment type that isn't an image must be audio. This leads to incorrect behavior when unsupported MIME types are passed. While upstream validation may prevent this in normal usage, the function should be defensive and either raise an error for unsupported types or explicitly handle only the types it supports.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -469,11 +469,14 @@ def _attachment(attachment):
                 "file_data": f"data:application/pdf;base64,{base64_content}",
             },
         }
-    if attachment.resolve_type().startswith("image/"):
+    elif attachment.resolve_type().startswith("image/"):
         return {"type": "image_url", "image_url": {"url": url}}
-    else:
+    elif attachment.resolve_type().startswith("audio/"):
         format_ = "wav" if attachment.resolve_type() == "audio/wav" else "mp3"
         return {
             "type": "input_audio",
             "input_audio": {
                 "data": base64_content,
                 "format": format_,
             },
         }
+    else:
+        raise ValueError(f"Unsupported attachment type: {attachment.resolve_type()}")
```