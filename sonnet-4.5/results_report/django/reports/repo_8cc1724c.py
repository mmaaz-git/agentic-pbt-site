import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(
        SECRET_KEY='test-secret-key',
        EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend',
        DEFAULT_FROM_EMAIL='test@example.com',
        EMAIL_SUBJECT_PREFIX='[Django] ',
        SERVER_EMAIL='root@localhost',
        ADMINS=[('Admin', 'admin@example.com')],
        MANAGERS=[('Manager', 'manager@example.com')],
    )

django.setup()

from django.core.mail import send_mail, mail_admins, mail_managers

# Test send_mail - should return 1
send_result = send_mail('Test Subject', 'Test Message', 'from@example.com', ['to@example.com'])
print(f"send_mail() returned: {send_result} (type: {type(send_result)})")

# Test mail_admins - should return 1 but returns None
admins_result = mail_admins('Admin Alert', 'Important message for admins')
print(f"mail_admins() returned: {admins_result} (type: {type(admins_result)})")

# Test mail_managers - should return 1 but returns None
managers_result = mail_managers('Manager Notice', 'Important message for managers')
print(f"mail_managers() returned: {managers_result} (type: {type(managers_result)})")