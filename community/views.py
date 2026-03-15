# community/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Count, Q
from django.contrib import messages
from .models import ForumCategory, ForumTopic, ForumReply, ForumLike
from django.utils import timezone


@login_required
def forum_home(request):
    """Main forum page showing all categories"""
    categories = ForumCategory.objects.annotate(
        topic_count=Count('topics'),
        reply_count=Count('topics__replies')
    ).all()
    
    # Get recent topics across all categories
    recent_topics = ForumTopic.objects.select_related('author', 'category').order_by('-created_at')[:5]
    
    # Get popular topics (most replies)
    popular_topics = ForumTopic.objects.annotate(
        reply_count=Count('replies')
    ).order_by('-reply_count')[:5]
    
    context = {
        'categories': categories,
        'recent_topics': recent_topics,
        'popular_topics': popular_topics,
    }
    return render(request, 'community/forum_home.html', context)


@login_required
def category_view(request, category_id):
    """View all topics in a category"""
    category = get_object_or_404(ForumCategory, id=category_id)
    topics = ForumTopic.objects.filter(category=category).select_related('author').annotate(
        reply_count=Count('replies')
    ).order_by('-is_pinned', '-updated_at')
    
    context = {
        'category': category,
        'topics': topics,
    }
    return render(request, 'community/category_view.html', context)


@login_required
def topic_view(request, topic_id):
    """View a topic and its replies"""
    topic = get_object_or_404(ForumTopic, id=topic_id)
    topic.increment_views()
    
    replies = topic.replies.select_related('author').all()
    
    # Check if user has liked the topic
    user_liked_topic = ForumLike.objects.filter(user=request.user, topic=topic).exists()
    
    # Get like counts for replies
    reply_likes = {}
    for reply in replies:
        reply_likes[reply.id] = {
            'count': reply.likes.count(),
            'user_liked': ForumLike.objects.filter(user=request.user, reply=reply).exists()
        }
    
    context = {
        'topic': topic,
        'replies': replies,
        'user_liked_topic': user_liked_topic,
        'reply_likes': reply_likes,
        'topic_like_count': topic.likes.count(),
    }
    return render(request, 'community/topic_view.html', context)


@login_required
def create_topic(request, category_id):
    """Create a new topic"""
    category = get_object_or_404(ForumCategory, id=category_id)
    
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        
        if not title or not content:
            messages.error(request, 'Title and content are required.')
            return redirect('category_view', category_id=category_id)
        
        topic = ForumTopic.objects.create(
            category=category,
            author=request.user,
            title=title,
            content=content
        )
        
        messages.success(request, 'Topic created successfully!')
        return redirect('topic_view', topic_id=topic.id)
    
    context = {'category': category}
    return render(request, 'community/create_topic.html', context)


@login_required
def create_reply(request, topic_id):
    """Create a reply to a topic"""
    topic = get_object_or_404(ForumTopic, id=topic_id)
    
    if topic.is_locked:
        messages.error(request, 'This topic is locked and cannot receive new replies.')
        return redirect('topic_view', topic_id=topic_id)
    
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        
        if not content:
            messages.error(request, 'Reply content cannot be empty.')
            return redirect('topic_view', topic_id=topic_id)
        
        ForumReply.objects.create(
            topic=topic,
            author=request.user,
            content=content
        )
        
        # Update topic's updated_at to bump it to the top
        topic.updated_at = timezone.now()
        topic.save()
        
        messages.success(request, 'Reply posted successfully!')
        return redirect('topic_view', topic_id=topic_id)
    
    return redirect('topic_view', topic_id=topic_id)


@login_required
def edit_topic(request, topic_id):
    """Edit a topic (only by author)"""
    topic = get_object_or_404(ForumTopic, id=topic_id)
    
    if topic.author != request.user:
        messages.error(request, 'You can only edit your own topics.')
        return redirect('topic_view', topic_id=topic_id)
    
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()
        content = request.POST.get('content', '').strip()
        
        if title and content:
            topic.title = title
            topic.content = content
            topic.save()
            messages.success(request, 'Topic updated successfully!')
            return redirect('topic_view', topic_id=topic_id)
    
    context = {'topic': topic}
    return render(request, 'community/edit_topic.html', context)


@login_required
def delete_topic(request, topic_id):
    """Delete a topic (only by author)"""
    topic = get_object_or_404(ForumTopic, id=topic_id)
    
    if topic.author != request.user:
        messages.error(request, 'You can only delete your own topics.')
        return redirect('topic_view', topic_id=topic_id)
    
    if request.method == 'POST':
        category_id = topic.category.id
        topic.delete()
        messages.success(request, 'Topic deleted successfully!')
        return redirect('category_view', category_id=category_id)
    
    return redirect('topic_view', topic_id=topic_id)


@login_required
def edit_reply(request, reply_id):
    """Edit a reply (only by author)"""
    reply = get_object_or_404(ForumReply, id=reply_id)
    
    if reply.author != request.user:
        messages.error(request, 'You can only edit your own replies.')
        return redirect('topic_view', topic_id=reply.topic.id)
    
    if request.method == 'POST':
        content = request.POST.get('content', '').strip()
        
        if content:
            reply.content = content
            reply.mark_edited()
            messages.success(request, 'Reply updated successfully!')
    
    return redirect('topic_view', topic_id=reply.topic.id)


@login_required
def delete_reply(request, reply_id):
    """Delete a reply (only by author)"""
    reply = get_object_or_404(ForumReply, id=reply_id)
    
    if reply.author != request.user:
        messages.error(request, 'You can only delete your own replies.')
        return redirect('topic_view', topic_id=reply.topic.id)
    
    if request.method == 'POST':
        topic_id = reply.topic.id
        reply.delete()
        messages.success(request, 'Reply deleted successfully!')
        return redirect('topic_view', topic_id=topic_id)
    
    return redirect('topic_view', topic_id=reply.topic.id)


@login_required
def toggle_like(request, content_type, content_id):
    """Toggle like on topic or reply"""
    if request.method == 'POST':
        if content_type == 'topic':
            topic = get_object_or_404(ForumTopic, id=content_id)
            like, created = ForumLike.objects.get_or_create(user=request.user, topic=topic)
            if not created:
                like.delete()
                liked = False
            else:
                liked = True
            like_count = topic.likes.count()
        elif content_type == 'reply':
            reply = get_object_or_404(ForumReply, id=content_id)
            like, created = ForumLike.objects.get_or_create(user=request.user, reply=reply)
            if not created:
                like.delete()
                liked = False
            else:
                liked = True
            like_count = reply.likes.count()
        else:
            return JsonResponse({'error': 'Invalid content type'}, status=400)
        
        return JsonResponse({
            'liked': liked,
            'like_count': like_count
        })
    
    return JsonResponse({'error': 'Invalid request'}, status=400)


@login_required
def search_forum(request):
    """Search topics and replies"""
    query = request.GET.get('q', '').strip()
    
    if not query:
        return redirect('forum_home')
    
    # Search in topics
    topics = ForumTopic.objects.filter(
        Q(title__icontains=query) | Q(content__icontains=query)
    ).select_related('author', 'category').annotate(
        reply_count=Count('replies')
    )[:20]
    
    context = {
        'query': query,
        'topics': topics,
    }
    return render(request, 'community/search_results.html', context)
