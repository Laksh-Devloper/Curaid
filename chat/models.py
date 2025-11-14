from django.db import models
from django.contrib.auth import get_user_model
import uuid


User = get_user_model()


class ChatSession(models.Model):
    user = models.ForeignKey('accounts.CustomUser', on_delete=models.CASCADE)
    session_id = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.user.email} - {self.title or 'Untitled Session'}"
    
    
class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    bot_response = models.TextField(blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages', null=True)  # ADD THIS LINE
    class Meta:
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.user.username}: {self.message}"


class UserTodo(models.Model):
    user = user = models.ForeignKey(User, on_delete=models.CASCADE)
    task_description = models.CharField(max_length=255)
    suggested_by_curo = models.BooleanField(default=True) # Was this suggested by the AI?
    created_at = models.DateTimeField(auto_now_add=True)
    due_date = models.DateField(null=True, blank=True) # For specific deadlines
    completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        status = " (Done)" if self.completed else ""
        return f"{self.user.username}: {self.task_description}{status}"

    class Meta:
        ordering = ['-created_at'] # Order by most recent first


