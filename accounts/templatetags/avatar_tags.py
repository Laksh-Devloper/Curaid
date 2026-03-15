# community/templatetags/avatar_tags.py
from django import template
import hashlib

register = template.Library()


@register.simple_tag
def avatar_url(user, style='adventurer', size=100):
    """
    Generate a 3D avatar URL using DiceBear Avatars API
    
    Styles available:
    - adventurer (3D characters)
    - avataaars (3D cartoon)
    - big-ears (3D cute)
    - bottts (3D robots)
    - lorelei (3D illustrated)
    - micah (3D minimal)
    - personas (3D people)
    """
    # Use username as seed for consistent avatars
    seed = user.username if hasattr(user, 'username') else str(user)
    
    # DiceBear API v7
    return f"https://api.dicebear.com/7.x/{style}/svg?seed={seed}&size={size}"


@register.simple_tag
def avatar_3d_url(user, size=100):
    """Generate a 3D-style avatar (adventurer style)"""
    return avatar_url(user, style='adventurer', size=size)


@register.simple_tag
def avatar_robot_url(user, size=100):
    """Generate a 3D robot avatar"""
    return avatar_url(user, style='bottts', size=size)


@register.simple_tag
def avatar_cartoon_url(user, size=100):
    """Generate a 3D cartoon avatar"""
    return avatar_url(user, style='avataaars', size=size)


@register.simple_tag
def gravatar_url(email, size=100, default='mp'):
    """
    Generate Gravatar URL as fallback
    default options: mp, identicon, monsterid, wavatar, retro, robohash
    """
    if not email:
        email = 'default@example.com'
    
    # Create MD5 hash of email
    email_hash = hashlib.md5(email.lower().encode('utf-8')).hexdigest()
    
    return f"https://www.gravatar.com/avatar/{email_hash}?s={size}&d={default}"


@register.filter
def avatar_color(username):
    """Generate a consistent color for username based on hash"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
        '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788',
        '#E63946', '#457B9D', '#F77F00', '#06FFA5', '#8338EC'
    ]
    
    # Hash username to get consistent color
    hash_value = sum(ord(c) for c in username)
    return colors[hash_value % len(colors)]


@register.filter
def initials(username):
    """Get user initials (first 2 letters)"""
    if not username:
        return 'U'
    
    parts = username.split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[1][0]).upper()
    return username[:2].upper()
