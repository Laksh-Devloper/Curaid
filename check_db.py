import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from community.models import ForumCategory, ForumTopic
print("Categories:", ForumCategory.objects.count())
print("Topics:", ForumTopic.objects.count())
