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

send_result = send_mail('Test', 'Message', 'from@example.com', ['to@example.com'])
print(f"send_mail() returned: {send_result}")
print(f"send_mail() return type: {type(send_result)}")

admins_result = mail_admins('Test', 'Message')
print(f"mail_admins() returned: {admins_result}")
print(f"mail_admins() return type: {type(admins_result)}")

managers_result = mail_managers('Test', 'Message')
print(f"mail_managers() returned: {managers_result}")
print(f"mail_managers() return type: {type(managers_result)}")