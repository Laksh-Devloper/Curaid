# community/management/commands/create_forum_categories.py
from django.core.management.base import BaseCommand
from community.models import ForumCategory


class Command(BaseCommand):
    help = 'Creates default forum categories'

    def handle(self, *args, **options):
        categories = [
            {
                'name': 'General Health',
                'description': 'Discuss general health topics, wellness tips, and healthy living',
                'icon': 'fa-heartbeat',
                'color': '#4A90E2',
                'order': 1
            },
            {
                'name': 'Mental Health',
                'description': 'Share experiences and support for mental health and emotional wellbeing',
                'icon': 'fa-brain',
                'color': '#9B59B6',
                'order': 2
            },
            {
                'name': 'Nutrition & Diet',
                'description': 'Recipes, meal plans, and nutrition advice',
                'icon': 'fa-apple-alt',
                'color': '#27AE60',
                'order': 3
            },
            {
                'name': 'Fitness & Exercise',
                'description': 'Workout routines, fitness goals, and exercise tips',
                'icon': 'fa-dumbbell',
                'color': '#E74C3C',
                'order': 4
            },
            {
                'name': 'Chronic Conditions',
                'description': 'Support and information for managing chronic health conditions',
                'icon': 'fa-notes-medical',
                'color': '#F39C12',
                'order': 5
            },
            {
                'name': 'Success Stories',
                'description': 'Share your health journey victories and inspire others',
                'icon': 'fa-trophy',
                'color': '#FFD700',
                'order': 6
            },
        ]

        created_count = 0
        for cat_data in categories:
            category, created = ForumCategory.objects.get_or_create(
                name=cat_data['name'],
                defaults=cat_data
            )
            if created:
                created_count += 1
                self.stdout.write(self.style.SUCCESS(f'Created category: {category.name}'))
            else:
                self.stdout.write(self.style.WARNING(f'Category already exists: {category.name}'))

        self.stdout.write(self.style.SUCCESS(f'\nTotal categories created: {created_count}'))
