# community/management/commands/populate_forum_quick.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from community.models import ForumCategory, ForumTopic, ForumReply
import random

User = get_user_model()


class Command(BaseCommand):
    help = 'Quickly populates forum with sample content (no AI required)'

    def handle(self, *args, **options):
        # Sample topics by category
        sample_data = {
            'General Health': [
                {
                    'title': 'Best morning routine for overall health?',
                    'content': 'I\'ve been trying to establish a better morning routine. Currently I wake up, check my phone, and rush to work. What does your ideal healthy morning look like? I\'d love to hear what works for you!',
                    'replies': [
                        'I start with 10 minutes of stretching and a glass of water. Game changer!',
                        'Meditation for 5 minutes really helps me stay calm throughout the day.',
                        'I do a quick 15-minute workout before breakfast. Energizes me for hours!',
                    ]
                },
                {
                    'title': 'How often should I get health checkups?',
                    'content': 'I\'m 28 years old and generally healthy. How often should I be seeing my doctor for checkups? I want to stay on top of my health but don\'t want to overdo it.',
                    'replies': [
                        'Annual checkups are usually recommended. Your doctor can advise based on your health history.',
                        'I go once a year for a physical. It\'s worth it for peace of mind!',
                    ]
                },
            ],
            'Mental Health': [
                {
                    'title': 'Dealing with work stress - your tips?',
                    'content': 'Work has been incredibly stressful lately. I find myself thinking about it even on weekends. What are your go-to strategies for managing work-related stress?',
                    'replies': [
                        'Setting boundaries helped me. I don\'t check work emails after 6 PM anymore.',
                        'Deep breathing exercises during breaks make a huge difference for me.',
                        'I started journaling before bed. It helps me process the day and sleep better.',
                        'Regular exercise is my stress reliever. Even a 20-minute walk helps!',
                    ]
                },
                {
                    'title': 'Meditation apps - which one do you use?',
                    'content': 'I want to start meditating but feeling overwhelmed by all the apps out there. Which meditation app do you recommend for beginners?',
                    'replies': [
                        'Headspace is great for beginners! Very user-friendly.',
                        'I love Calm. The sleep stories are amazing too.',
                    ]
                },
            ],
            'Nutrition & Diet': [
                {
                    'title': 'Meal prep Sunday - share your recipes!',
                    'content': 'Starting meal prep this Sunday. Looking for healthy, easy recipes that last well in the fridge. What are your go-to meal prep recipes?',
                    'replies': [
                        'Chicken and veggie bowls are my staple. Easy to customize!',
                        'I make a big batch of quinoa salad with chickpeas. Lasts 4-5 days.',
                        'Overnight oats for breakfast! So many flavor combinations possible.',
                    ]
                },
                {
                    'title': 'Protein intake - am I getting enough?',
                    'content': 'I\'m trying to build muscle but not sure if I\'m eating enough protein. I\'m 70kg and workout 4x a week. How much protein should I aim for daily?',
                    'replies': [
                        'General rule is 1.6-2.2g per kg of body weight for muscle building.',
                        'Track your intake for a week to see where you\'re at. MyFitnessPal helps!',
                    ]
                },
            ],
            'Fitness & Exercise': [
                {
                    'title': 'Beginner gym routine - where to start?',
                    'content': 'Just joined a gym but feeling lost. What\'s a good beginner routine for someone who hasn\'t worked out in years? My goal is general fitness and weight loss.',
                    'replies': [
                        'Start with full-body workouts 3x a week. Focus on compound movements!',
                        'Don\'t skip cardio! 20-30 minutes after weights is perfect.',
                        'Get a trainer for the first month if you can. Proper form is crucial!',
                        'Be patient with yourself. Progress takes time but it\'s so worth it!',
                    ]
                },
                {
                    'title': 'Running vs. cycling for cardio?',
                    'content': 'I need to improve my cardio but can\'t decide between running and cycling. Which do you prefer and why?',
                    'replies': [
                        'Cycling is easier on the joints. I switched from running and love it!',
                        'Running is more convenient - just need shoes! But cycling is fun too.',
                    ]
                },
            ],
            'Chronic Conditions': [
                {
                    'title': 'Managing diabetes - daily routine tips',
                    'content': 'Recently diagnosed with Type 2 diabetes. Still learning how to manage it. What does your daily routine look like? Any tips for a newbie?',
                    'replies': [
                        'Consistent meal times really help with blood sugar management.',
                        'I check my levels before and after meals. Helps me understand what foods work.',
                        'Walking after meals has been a game-changer for my numbers!',
                    ]
                },
            ],
            'Success Stories': [
                {
                    'title': 'Lost 20kg in 6 months - my journey',
                    'content': 'I\'m so proud to share that I\'ve lost 20kg over the past 6 months! It wasn\'t easy but it was worth it. Happy to answer questions about my journey!',
                    'replies': [
                        'Congratulations! That\'s amazing progress! What was your biggest challenge?',
                        'Wow, inspiring! Did you follow a specific diet plan?',
                        'This gives me hope! I\'m just starting my journey.',
                    ]
                },
                {
                    'title': 'Finally ran my first 5K!',
                    'content': 'Completed my first 5K race today! Six months ago I couldn\'t run for 5 minutes. If you\'re thinking about starting, just do it!',
                    'replies': [
                        'Congratulations! That\'s a huge milestone!',
                        'Amazing! What training program did you follow?',
                    ]
                },
            ],
        }

        # Get bot users
        bot_users = list(User.objects.filter(email__contains='curaid-bot.com'))
        
        if not bot_users:
            self.stdout.write(self.style.ERROR('No bot users found! Run: python manage.py create_forum_bots'))
            return

        topics_created = 0
        replies_created = 0

        # Create content for each category
        for category_name, topics_data in sample_data.items():
            try:
                category = ForumCategory.objects.get(name=category_name)
                
                for topic_data in topics_data:
                    # Random bot as author
                    author = random.choice(bot_users)
                    
                    # Create topic
                    topic = ForumTopic.objects.create(
                        category=category,
                        author=author,
                        title=topic_data['title'],
                        content=topic_data['content']
                    )
                    topics_created += 1
                    self.stdout.write(self.style.SUCCESS(f'✓ Created: {topic.title}'))
                    
                    # Create replies
                    for reply_text in topic_data['replies']:
                        reply_author = random.choice([u for u in bot_users if u != author])
                        ForumReply.objects.create(
                            topic=topic,
                            author=reply_author,
                            content=reply_text
                        )
                        replies_created += 1
                    
                    self.stdout.write(f'  Added {len(topic_data["replies"])} replies')
                    
            except ForumCategory.DoesNotExist:
                self.stdout.write(self.style.WARNING(f'Category not found: {category_name}'))
                continue

        self.stdout.write(self.style.SUCCESS(f'\n✅ Forum populated!'))
        self.stdout.write(self.style.SUCCESS(f'Topics created: {topics_created}'))
        self.stdout.write(self.style.SUCCESS(f'Replies created: {replies_created}'))
