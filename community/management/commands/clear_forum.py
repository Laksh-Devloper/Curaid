# community/management/commands/clear_forum.py
from django.core.management.base import BaseCommand
from community.models import ForumTopic, ForumReply, ForumLike

class Command(BaseCommand):
    help = 'Clears all forum content (topics, replies, likes)'

    def handle(self, *args, **options):
        # Count before deletion
        topics_count = ForumTopic.objects.count()
        replies_count = ForumReply.objects.count()
        likes_count = ForumLike.objects.count()

        # Delete all
        ForumLike.objects.all().delete()
        ForumReply.objects.all().delete()
        ForumTopic.objects.all().delete()

        self.stdout.write(self.style.SUCCESS(f'✅ Forum cleared!'))
        self.stdout.write(self.style.SUCCESS(f'Deleted {topics_count} topics'))
        self.stdout.write(self.style.SUCCESS(f'Deleted {replies_count} replies'))
        self.stdout.write(self.style.SUCCESS(f'Deleted {likes_count} likes'))
