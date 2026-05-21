import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Without config
model1 = genai.GenerativeModel(model_name="gemini-2.5-flash")
res1 = model1.generate_content("Write a 100 word story.")
print("Without config:", len(res1.text.split()), "words")

# With large max_output_tokens
model2 = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={"max_output_tokens": 2048}
)
res2 = model2.generate_content("Write a 100 word story.")
print("With 2048:", len(res2.text.split()), "words")
