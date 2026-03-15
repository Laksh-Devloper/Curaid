from django.contrib import admin
from .models import ChatMessage, ChatSession, UserTodo

# Register ChatMessage with a basic admin interface
@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('user', 'session', 'message', 'bot_response', 'timestamp')  # Fields to display in the list view
    list_filter = ('user', 'timestamp', 'session')  # Add filters for user and timestamp
    search_fields = ('message', 'bot_response')  # Enable search by message and bot_response
    ordering = ('-timestamp',)

# Register ChatSession
@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('user', 'title', 'session_id', 'created_at', 'updated_at', 'is_active')
    list_filter = ('is_active', 'created_at', 'user')
    search_fields = ('title', 'user__email', 'user__username')
    readonly_fields = ('session_id', 'created_at', 'updated_at')
    ordering = ('-updated_at',)

# Register UserTodo
@admin.register(UserTodo)
class UserTodoAdmin(admin.ModelAdmin):
    list_display = ('user', 'task_description', 'suggested_by_curo', 'completed', 'due_date', 'created_at')
    list_filter = ('completed', 'suggested_by_curo', 'due_date', 'user')
    search_fields = ('task_description', 'user__email', 'user__username')
    readonly_fields = ('created_at', 'completed_at')
    ordering = ('-created_at',)

