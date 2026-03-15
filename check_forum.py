import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'case_companion.settings')
django.setup()

from community.models import ForumTopic, ForumReply

print(f"Total Topics: {ForumTopic.objects.count()}")
print(f"Total Replies: {ForumReply.objects.count()}")
print("\nSample topics with reply counts:")
for topic in ForumTopic.objects.all()[:5]:
    print(f"- {topic.title}: {topic.replies.count()} replies")
