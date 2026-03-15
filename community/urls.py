# community/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.forum_home, name='forum_home'),
    path('category/<int:category_id>/', views.category_view, name='category_view'),
    path('topic/<uuid:topic_id>/', views.topic_view, name='topic_view'),
    path('category/<int:category_id>/create/', views.create_topic, name='create_topic'),
    path('topic/<uuid:topic_id>/reply/', views.create_reply, name='create_reply'),
    path('topic/<uuid:topic_id>/edit/', views.edit_topic, name='edit_topic'),
    path('topic/<uuid:topic_id>/delete/', views.delete_topic, name='delete_topic'),
    path('reply/<uuid:reply_id>/edit/', views.edit_reply, name='edit_reply'),
    path('reply/<uuid:reply_id>/delete/', views.delete_reply, name='delete_reply'),
    path('like/<str:content_type>/<uuid:content_id>/', views.toggle_like, name='toggle_like'),
    path('search/', views.search_forum, name='search_forum'),
]
