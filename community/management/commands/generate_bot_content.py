# community/management/commands/generate_bot_content.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from community.models import ForumCategory, ForumTopic, ForumReply
import google.generativeai as genai
from django.conf import settings
import random
import time

User = get_user_model()


class Command(BaseCommand):
    help = 'Generates AI-powered forum content from bots'

    def add_arguments(self, parser):
        parser.add_argument(
            '--topics',
            type=int,
            default=10,
            help='Number of topics to create'
        )
        parser.add_argument(
            '--replies',
            type=int,
            default=3,
            help='Average number of replies per topic'
        )

    def handle(self, *args, **options):
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Get bot users
        bot_users = User.objects.filter(username__endswith='_AI') | User.objects.filter(
            email__contains='curaid-bot.com'
        )
        
        if not bot_users.exists():
            self.stdout.write(self.style.ERROR('No bot users found! Run: python manage.py create_forum_bots'))
            return

        # Get all categories
        categories = ForumCategory.objects.all()
        if not categories.exists():
            self.stdout.write(self.style.ERROR('No categories found! Run: python manage.py create_forum_categories'))
            return

        num_topics = options['topics']
        avg_replies = options['replies']

        self.stdout.write(self.style.SUCCESS(f'Generating {num_topics} topics with ~{avg_replies} replies each...'))

        topics_created = 0
        replies_created = 0

        for i in range(num_topics):
            try:
                # Select random category and bot
                category = random.choice(categories)
                author = random.choice(bot_users)

                # Generate topic using AI
                topic_prompt = f"""Generate a realistic forum topic for a health community in the category: {category.name}

Category description: {category.description}

Create a topic with:
1. A catchy, engaging title (max 100 characters)
2. Detailed content (2-3 paragraphs, 100-200 words)

The topic should be:
- Helpful and informative
- Personal but professional
- Encouraging discussion
- Related to {category.name.lower()}

Format your response as:
TITLE: [title here]
CONTENT: [content here]
"""

                response = model.generate_content(topic_prompt)
                content = response.text

                # Parse response
                if 'TITLE:' in content and 'CONTENT:' in content:
                    title = content.split('TITLE:')[1].split('CONTENT:')[0].strip()
                    body = content.split('CONTENT:')[1].strip()

                    # Create topic
                    topic = ForumTopic.objects.create(
                        category=category,
                        author=author,
                        title=title[:200],  # Limit to max length
                        content=body
                    )
                    topics_created += 1
                    self.stdout.write(self.style.SUCCESS(f'✓ Created topic: {title[:50]}...'))

                    # Generate replies
                    num_replies = random.randint(max(1, avg_replies - 2), avg_replies + 2)
                    
                    for j in range(num_replies):
                        try:
                            reply_author = random.choice([u for u in bot_users if u != author])
                            
                            reply_prompt = f"""Generate a helpful reply to this forum topic:

Title: {title}
Content: {body[:200]}...

Create a reply that:
- Is supportive and encouraging
- Adds value to the discussion
- Is 2-4 sentences (50-100 words)
- Sounds natural and conversational
- May share personal experience or tips

Just write the reply content, no labels or formatting."""

                            reply_response = model.generate_content(reply_prompt)
                            reply_content = reply_response.text.strip()

                            # Create reply
                            ForumReply.objects.create(
                                topic=topic,
                                author=reply_author,
                                content=reply_content
                            )
                            replies_created += 1
                            
                            # Small delay to avoid rate limiting
                            time.sleep(0.5)
                            
                        except Exception as e:
                            self.stdout.write(self.style.WARNING(f'  Failed to create reply: {str(e)}'))
                            continue

                    self.stdout.write(self.style.SUCCESS(f'  Added {num_replies} replies'))

                # Delay between topics
                time.sleep(1)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Failed to create topic: {str(e)}'))
                continue

        self.stdout.write(self.style.SUCCESS(f'\n✅ Generation complete!'))
        self.stdout.write(self.style.SUCCESS(f'Topics created: {topics_created}'))
        self.stdout.write(self.style.SUCCESS(f'Replies created: {replies_created}'))
