import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from chat.views import _generate_ai_content

inputs = {'pregnancies': 0.0, 'glucose': 120.0, 'blood_pressure': 90.0, 'skin_thickness': 20.0, 'insulin': 90.0, 'bmi': 23.0, 'diabetes_pedigree_function': 0.9, 'age': 45.0}
prediction_text = 'healthy (no diabetes risk)'
prompt = (
    f"You are Cura Sphere, a medical AI assistant. The user has provided their medical parameters: {inputs}. "
    f"Our internal ML model predicts they are '{prediction_text}'. "
    "Please write a comprehensive, empathetic, and detailed response to the user. "
    "Explain what their key parameters mean (e.g. glucose, BMI, insulin) in relation to the prediction, and offer actionable lifestyle advice. "
    "Use markdown for bolding and bullet points to make it easy to read. Keep it strictly related to diabetes prevention/management."
)

print("Prompting Gemini...")
res = _generate_ai_content(prompt)
print("RESPONSE:\n", res)
