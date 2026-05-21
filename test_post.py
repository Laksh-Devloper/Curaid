import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "case_companion.settings")
django.setup()

from community.models import ForumCategory, ForumTopic
from accounts.models import CustomUser

user = CustomUser.objects.first()
category = ForumCategory.objects.first()

if user and category:
    topic = ForumTopic.objects.create(
        category=category,
        author=user,
        title="Test Topic",
        content="This is a test topic"
    )
    print("Created topic:", topic.id)
    print("Total topics in DB:", ForumTopic.objects.count())
else:
    print("Missing user or category")
