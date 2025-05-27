import os
import django
from django.core.management import call_command

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'weapon_detection_app.settings')
django.setup()

try:
    call_command('migrate')
    print("✅ Migrations completed successfully.")
except Exception as e:
    print(f"⚠️ Migration failed: {e}")
    