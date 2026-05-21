import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from chat.views import GEMINI_MODEL

print("Prompting Gemini to write a 500 word story...")
try:
    response = GEMINI_MODEL.generate_content("Write a 500 word story about a brave knight.")
    print("Length of response:", len(response.text.split()))
    print("Response snippet:", response.text[:200])
except Exception as e:
    print("Error:", e)
