import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from chat.views import handle_disease_predictor

input_data = "pregnancies: 0, glucose: 120, blood_pressure: 90, skin_thickness: 20, insulin: 90, bmi: 23, diabetes_pedigree_function: 0.9, age: 45"
print("Calling handle_disease_predictor...")
res = handle_disease_predictor(input_data, "diabetes")
print("RESULT:")
print(repr(res))
