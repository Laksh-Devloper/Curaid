# community/admin.py
from django.contrib import admin
from .models import ForumCategory, ForumTopic, ForumReply, ForumLike


@admin.register(ForumCategory)
class ForumCategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'order', 'topic_count', 'post_count', 'created_at']
    list_editable = ['order']
    search_fields = ['name', 'description']
    ordering = ['order', 'name']


@admin.register(ForumTopic)
class ForumTopicAdmin(admin.ModelAdmin):
    list_display = ['title', 'author', 'category', 'is_pinned', 'is_locked', 'views', 'created_at']
    list_filter = ['category', 'is_pinned', 'is_locked', 'created_at']
    search_fields = ['title', 'content', 'author__username']
    list_editable = ['is_pinned', 'is_locked']
    readonly_fields = ['id', 'views', 'created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(ForumReply)
class ForumReplyAdmin(admin.ModelAdmin):
    list_display = ['topic', 'author', 'is_edited', 'created_at']
    list_filter = ['is_edited', 'created_at']
    search_fields = ['content', 'author__username', 'topic__title']
    readonly_fields = ['id', 'created_at', 'updated_at']
    ordering = ['-created_at']


@admin.register(ForumLike)
class ForumLikeAdmin(admin.ModelAdmin):
    list_display = ['user', 'topic', 'reply', 'created_at']
    list_filter = ['created_at']
    search_fields = ['user__username']
    ordering = ['-created_at']
