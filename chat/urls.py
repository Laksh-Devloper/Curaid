from django.urls import path
from . import views

urlpatterns = [
    path('', views.chat_room, name='chat_room'),  # Matches 'chat_room'
    path('generate_health_report/', views.generate_health_report, name='generate_health_report'),
     path('mark_todo_done/<int:todo_id>/', views.mark_todo_done, name='mark_todo_done'),
    path('clear_completed_todos/', views.clear_completed_todos, name='clear_completed_todos'),
    path('health-dashboard/', views.health_dashboard_view, name='health_dashboard'),
    path('new-session/', views.new_chat_session, name='new_chat_session'),
path('load-session/<uuid:session_id>/', views.load_chat_session, name='load_chat_session'),
path('get-sessions/', views.get_chat_sessions, name='get_chat_sessions'),
path('delete-session/<uuid:session_id>/', views.delete_chat_session, name='delete_chat_session'),
]