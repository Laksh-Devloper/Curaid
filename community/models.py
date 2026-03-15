# community/models.py
from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
import uuid

User = get_user_model()


class ForumCategory(models.Model):
    """Categories for organizing forum topics"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, default='fa-comments')  # FontAwesome icon class
    color = models.CharField(max_length=7, default='#3AAFA9')  # Hex color
    created_at = models.DateTimeField(auto_now_add=True)
    order = models.IntegerField(default=0)  # For custom ordering
    
    class Meta:
        verbose_name_plural = 'Forum Categories'
        ordering = ['order', 'name']
    
    def __str__(self):
        return self.name
    
    def topic_count(self):
        return self.topics.count()
    
    def post_count(self):
        return sum(topic.replies.count() + 1 for topic in self.topics.all())


class ForumTopic(models.Model):
    """Discussion topics created by users"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    category = models.ForeignKey(ForumCategory, on_delete=models.CASCADE, related_name='topics')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='forum_topics')
    title = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_pinned = models.BooleanField(default=False)  # Pinned topics appear first
    is_locked = models.BooleanField(default=False)  # Locked topics can't receive replies
    views = models.IntegerField(default=0)
    
    class Meta:
        ordering = ['-is_pinned', '-updated_at']
    
    def __str__(self):
        return self.title
    
    def reply_count(self):
        return self.replies.count()
    
    def last_activity(self):
        last_reply = self.replies.order_by('-created_at').first()
        if last_reply:
            return last_reply.created_at
        return self.created_at
    
    def increment_views(self):
        self.views += 1
        self.save(update_fields=['views'])


class ForumReply(models.Model):
    """Replies to forum topics"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    topic = models.ForeignKey(ForumTopic, on_delete=models.CASCADE, related_name='replies')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='forum_replies')
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_edited = models.BooleanField(default=False)
    
    class Meta:
        verbose_name_plural = 'Forum Replies'
        ordering = ['created_at']
    
    def __str__(self):
        return f"Reply by {self.author.username} on {self.topic.title}"
    
    def mark_edited(self):
        self.is_edited = True
        self.save(update_fields=['is_edited', 'updated_at'])


class ForumLike(models.Model):
    """Likes for topics and replies"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    topic = models.ForeignKey(ForumTopic, on_delete=models.CASCADE, null=True, blank=True, related_name='likes')
    reply = models.ForeignKey(ForumReply, on_delete=models.CASCADE, null=True, blank=True, related_name='likes')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = [['user', 'topic'], ['user', 'reply']]
    
    def __str__(self):
        if self.topic:
            return f"{self.user.username} likes topic: {self.topic.title}"
        return f"{self.user.username} likes reply"
