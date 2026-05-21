# community/management/commands/create_forum_bots.py
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from community.models import ForumCategory, ForumTopic, ForumReply
import random

User = get_user_model()


class Command(BaseCommand):
    help = 'Creates bot users for the community forum'

    def handle(self, *args, **options):
        # Bot usernames with health/wellness theme
        bot_names = [
            'HealthyHabits_AI',
            'WellnessWarrior',
            'FitnessFanatic',
            'NutritionNinja',
            'MindfulMover',
            'YogaYoda',
            'CardioKing',
            'ProteinPro',
            'SleepSpecialist',
            'HydrationHero',
            'StretchMaster',
            'CalorieCounter',
            'MeditationMentor',
            'VitaminVanguard',
            'WorkoutWizard',
        ]

        created_count = 0
        
        for bot_name in bot_names:
            # Create bot user if doesn't exist
            user, created = User.objects.get_or_create(
                username=bot_name,
                defaults={
                    'email': f'{bot_name.lower()}@cura-sphere-bot.com',
                    'is_active': True,
                }
            )
            
            if created:
                # Set a random password (bots won't login)
                user.set_password('bot_password_' + str(random.randint(1000, 9999)))
                user.save()
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'Created bot: {bot_name}'))
            else:
                self.stdout.write(self.style.WARNING(f'Bot already exists: {bot_name}'))

        self.stdout.write(self.style.SUCCESS(f'\nTotal bots created: {created_count}'))
        self.stdout.write(self.style.SUCCESS(f'Total bots in system: {len(bot_names)}'))
