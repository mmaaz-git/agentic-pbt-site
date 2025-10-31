# Bug Report: django.core.cache.backends.base.BaseCache.incr_version Deletes Cache Entry When delta=0

**Target**: `django.core.cache.backends.base.BaseCache.incr_version`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Calling `incr_version(key, delta=0)` unexpectedly deletes the cached value instead of being a no-op, causing silent data loss in the cache.

## Property-Based Test

```python
import sys
from hypothesis import given, strategies as st

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache


@given(st.text(min_size=1), st.integers(), st.integers(min_value=1, max_value=100))
def test_incr_version_zero_delta_deletes_key(key, value, version):
    """Property: incr_version with delta=0 should not delete the key."""
    cache = LocMemCache(f"test_{version}", {"timeout": 300})
    cache.clear()

    cache.set(key, value, version=version)
    assert cache.has_key(key, version=version), "Key should exist before incr_version"

    new_version = cache.incr_version(key, delta=0, version=version)
    assert new_version == version, f"Version should remain {version}, got {new_version}"

    assert cache.has_key(key, version=version), \
        f"Key should still exist at version {version} after incr_version with delta=0"


if __name__ == "__main__":
    test_incr_version_zero_delta_deletes_key()
```

<details>

<summary>
**Failing input**: `key='0', value=0, version=1`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x96\x9eF÷0hn\U000ac518\U000c931d³\x13𩖾\x07'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:à(\x99o\x06\U000771a8¼Á\xa0\U000d1161`À\x94'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:#¼ª±z¯\x90\t'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x82\x02'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U000e1efd\x96¹_rj T©_9'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U00085837µ\x13'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:7Û@Ó\x14'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:vAc\U0009e060\x01\ue6c4\U000387a6\U000372f3¦Ì\U000e8cd7\U00052612L'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U0005aa31êò\x13'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:¾/f\U000e8afe\U000f098dz\x8bÁM\x93úÆ\x8a[tÖ\x96ÅLg\x05Òï⌨y\x8a\U000dd51foÛxã\x82`\r\U00077412õ,ÿ@W/\U000975b1Ð-J\x0e\n;ø\x8c'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:î\U000bbc64\U00105280𧅗\U000dec1aÁ\x1cô'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:ÎJ\x00'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ":1:ò\U00109a81\x8do\x10Õ\x1d掭'"
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x1fåü\x081\U000ef603n\U000f33d4ß¹\x05\x90öC'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:x '
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:ꁞF\U000e6af6\x1b\U001092d8\x01\r'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:öÓ\x1ab'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:¾¿\x05´1\U00098b91\x11\U000ba6ab@©ÑÑ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:`\x18ûÊ\x90'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x80䶸&Ý\U000d8421§\U000a8e8b\x05\x02'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U000e081e\x08\x8dÎ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x11cò'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:þ\nO\x1f'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x02ì$\x9ca'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:Í\x8aü1\x017x\x83x+:\x9e\x12\U00078d75+\U001089f9+\U00090a42\U000b79a9\x02\t¿T¬\U000329d8Ç\x7fÞè\x84¦\U000bc380¡ßê'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x00°\U00048ac1«¢\U0004469a'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:]û)\x8e\x0f'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:3Â\x94S\x1câ¥\x96I\x07\x9c'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x05îð='
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:]\U00049929\x08\U00037b66w\U000d53a0\x1b'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:|\x1e3åÒ\x9c¤\x14ûq'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:+ËÖ`4áÄ{é9\U0008d365l-\U000cc746\x9fÖ2\x11f2´\x88SÈ8'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x0eµÈ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U000b5ee0\U0007eb72\x9f¹æ\x1c\x95ù\U000de478µ\U000f8565'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:F\x00\U0001eefdÜ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\U0008c263𮹘\x18åS\U000782db'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:;æ/\U000f99be\x9e㛫ÏaO\x192\U001011f1\x97\U000ec0d1#\U00045ba6'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x9f\x14\U00033916'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x0e\U000caa5d\U000bf7b3'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:\x03Ð\xa0>'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:ìw\x06\r?k'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:];\x16½'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:×\U0005ab72)çãÒ\x1b'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':1:r\U0001b823\U000c9eae\x0f\x83\x85\x97h2'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':24:|\x1e3åÒ\x9c¤\x14ûq'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':33:\U000b5ee0\U0007eb72\x9f¹æ\x1c\x95ù\U000de478µ\U000f8565'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':26:\x00°\U00048ac1«¢\U0004469a'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':72:\U000e1efd\x96¹_rj T©_9'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':4:à(\x99o\x06\U000771a8¼Á\xa0\U000d1161`À\x94'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':80:\x96\x9eF÷0hn\U000ac518\U000c931d³\x13𩖾\x07'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':63:\x11cò'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':52:\x02ì$\x9ca'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':32:]\U00049929\x08\U00037b66w\U000d53a0\x1b'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':4:×\U0005ab72)çãÒ\x1b'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':67:\x02ì$\x9ca'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':17:7Û@Ó\x14'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':16:ìw\x06\r?k'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':82:\x02ì$\x9ca'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':25:]û)\x8e\x0f'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':65:\x0eµÈ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':76:\x1fåü\x081\U000ef603n\U000f33d4ß¹\x05\x90öC'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ":79:ò\U00109a81\x8do\x10Õ\x1d掭'"
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':50:F\x00\U0001eefdÜ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':64:7Û@Ó\x14'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':59:]\U00049929\x08\U00037b66w\U000d53a0\x1b'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':7:à(\x99o\x06\U000771a8¼Á\xa0\U000d1161`À\x94'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':16:7Û@Ó\x14'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':17:3Â\x94S\x1câ¥\x96I\x07\x9c'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':16:\U0008c263𮹘\x18åS\U000782db'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':79:\U00085837µ\x13'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':71:\x9f\x14\U00033916'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':48:\U000e081e\x08\x8dÎ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':93:];\x16½'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':11:Í\x8aü1\x017x\x83x+:\x9e\x12\U00078d75+\U001089f9+\U00090a42\U000b79a9\x02\t¿T¬\U000329d8Ç\x7fÞè\x84¦\U000bc380¡ßê'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':66:\x0e\U000caa5d\U000bf7b3'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':96:öÓ\x1ab'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':3:\x82\x02'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':85:\U000e1efd\x96¹_rj T©_9'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':13:ÎJ\x00'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':35:\U0005aa31êò\x13'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':99:`\x18ûÊ\x90'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':93:\U000b5ee0\U0007eb72\x9f¹æ\x1c\x95ù\U000de478µ\U000f8565'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':78:#¼ª±z¯\x90\t'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':32:\x0eµÈ'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':96:\x9f\x14\U00033916'
  warnings.warn(warning, CacheKeyWarning)
/home/npc/miniconda/lib/python3.13/site-packages/django/core/cache/backends/base.py:119: CacheKeyWarning: Cache key contains characters that will cause errors if used with memcached: ':11:ìw\x06\r?k'
  warnings.warn(warning, CacheKeyWarning)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 26, in <module>
    test_incr_version_zero_delta_deletes_key()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 10, in test_incr_version_zero_delta_deletes_key
    def test_incr_version_zero_delta_deletes_key(key, value, version):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 21, in test_incr_version_zero_delta_deletes_key
    assert cache.has_key(key, version=version), \
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Key should still exist at version 1 after incr_version with delta=0
Falsifying example: test_incr_version_zero_delta_deletes_key(
    # The test always failed when commented parts were varied together.
    key='0',  # or any other generated value
    value=0,  # or any other generated value
    version=1,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.cache.backends.locmem import LocMemCache

cache = LocMemCache("test", {"timeout": 300})
cache.clear()

# Set a key-value pair at version 1
cache.set("mykey", "myvalue", version=1)
print(f"Before incr_version: has_key('mykey', version=1) = {cache.has_key('mykey', version=1)}")
print(f"Before incr_version: get('mykey', version=1) = {cache.get('mykey', version=1)}")

# Call incr_version with delta=0 (should be a no-op)
print("\nCalling incr_version('mykey', delta=0, version=1)...")
new_version = cache.incr_version("mykey", delta=0, version=1)
print(f"Returned version: {new_version}")

# Check if the key still exists
print(f"\nAfter incr_version: has_key('mykey', version=1) = {cache.has_key('mykey', version=1)}")
print(f"After incr_version: get('mykey', version=1) = {cache.get('mykey', version=1)}")
```

<details>

<summary>
Output demonstrates that the cached value is deleted
</summary>
```
Before incr_version: has_key('mykey', version=1) = True
Before incr_version: get('mykey', version=1) = myvalue

Calling incr_version('mykey', delta=0, version=1)...
Returned version: 1

After incr_version: has_key('mykey', version=1) = False
After incr_version: get('mykey', version=1) = None
```
</details>

## Why This Is A Bug

The `incr_version` method in Django's cache backend contains a logic error that causes unexpected data deletion when `delta=0`. The method is documented to "Add delta to the cache version for the supplied key" and return the new version. However, when `delta=0`, instead of being a no-op operation (as would be expected when incrementing by zero), the method deletes the cached value entirely.

The implementation flaw occurs in the method's logic (lines 346-360 of `/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages/django/core/cache/backends/base.py`):

1. Line 354: Retrieves the value from `version`
2. Line 358: Sets the value at `version + delta`
3. Line 359: Deletes the value from `version`

When `delta=0`, both the set operation (step 2) and the delete operation (step 3) target the same version. This means the value is first written to version `v` and then immediately deleted from version `v`, resulting in complete data loss.

This violates several principles:
- **Principle of Least Surprise**: No reasonable developer would expect that incrementing a version by 0 would delete their data
- **Mathematical Consistency**: Adding 0 to any number should leave it unchanged (identity element property)
- **Data Integrity**: Silent data deletion without any warning or error is a severe side effect

The same issue affects `decr_version(key, delta=0)` since it internally calls `incr_version(key, -0)`, which is still 0.

## Relevant Context

This bug could occur in production scenarios where `delta` is calculated dynamically. For example:
- Conditional version bumping where sometimes no bump is needed
- Configuration-driven delta values that might be set to 0
- Mathematical calculations that occasionally produce 0 as the delta

The Django documentation for `incr_version` states it should "Add delta to the cache version for the supplied key" but doesn't specify behavior for `delta=0`. The implementation should either:
1. Treat `delta=0` as a no-op (preferred solution)
2. Raise a `ValueError` if `delta=0` is not supported

Documentation link: https://docs.djangoproject.com/en/stable/topics/cache/#cache-versioning

The async version `aincr_version` has the same bug (lines 362-373).

## Proposed Fix

```diff
--- a/django/core/cache/backends/base.py
+++ b/django/core/cache/backends/base.py
@@ -346,6 +346,9 @@ class BaseCache:
     def incr_version(self, key, delta=1, version=None):
         """
         Add delta to the cache version for the supplied key. Return the new
         version.
         """
+        if delta == 0:
+            # No-op: return current version without modifying cache
+            return self.version if version is None else version
         if version is None:
             version = self.version

@@ -361,6 +364,9 @@ class BaseCache:

     async def aincr_version(self, key, delta=1, version=None):
         """See incr_version()."""
+        if delta == 0:
+            # No-op: return current version without modifying cache
+            return self.version if version is None else version
         if version is None:
             version = self.version
```