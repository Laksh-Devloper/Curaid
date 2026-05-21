# accounts/views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.conf import settings
from django.http import HttpResponse, HttpRequest
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView
from google.oauth2 import id_token
from google.auth.transport import requests
import uuid
from .models import CustomUser


def index_view(request):
    return render(request, 'index.html')


def google_login(request):
    return render(request, 'google_login.html')


def profile_view(request):
    if request.user.is_authenticated:
        return render(request, 'profile.html', {'user': request.user})
    return redirect('login')


@csrf_exempt
def auth_receiver(request):
    """Handle Google OAuth authentication"""
    if request.method == 'POST':
        token = request.POST.get('credential')
        try:
            user_data = id_token.verify_oauth2_token(
                token, requests.Request(), settings.GOOGLE_OAUTH_CLIENT_ID
            )
            print(f"Google User Data: {user_data}")
            email = user_data['email']
            username = user_data.get('given_name', email.split('@')[0])
            
            user, created = CustomUser.objects.get_or_create(
                email=email,
                defaults={
                    'username': username[:30],
                }
            )
            
            if created:
                user.set_unusable_password()
                user.save()
            
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            print(f"Logged in user: {user.username}")
            return redirect('profile')
            
        except ValueError as e:
            print(f"Google token error: {e}")
            messages.error(request, 'Invalid Google authentication. Please try again.')
            return redirect('login')
    
    return HttpResponse("Method not allowed", status=405)


def signup_view(request):
    """Handle user registration - simple email/password"""
    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '').strip()
        
        # Validation
        if not name or not email or not password:
            messages.error(request, 'All fields are required.')
            return render(request, 'login.html', {'signup_errors': 'All fields are required.'})
        
        # Check if user already exists
        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, 'Email already registered. Please login instead.')
            return render(request, 'login.html', {'signup_errors': 'Email already registered.'})
        
        # Generate unique username from email
        base_username = email.split('@')[0]
        username = base_username[:30]
        
        # Make username unique if needed
        counter = 1
        while CustomUser.objects.filter(username=username).exists():
            username = f"{base_username[:24]}_{counter}"
            counter += 1
        
        try:
            # Create user
            user = CustomUser.objects.create(
                username=username,
                email=email,
                first_name=name.split()[0] if name else '',
                last_name=' '.join(name.split()[1:]) if len(name.split()) > 1 else ''
            )
            user.set_password(password)
            user.save()
            
            # Auto-login and redirect to profile with success notification
            login(request, user)
            messages.success(request, 'profile_created')
            return redirect('profile')
            
        except Exception as e:
            print(f"Signup error: {e}")
            messages.error(request, f'An error occurred during signup. Please try again.')
            return render(request, 'login.html', {'signup_errors': 'An error occurred. Please try again.'})
    
    return render(request, 'login.html')


def login_view(request):
    """Handle user login - accepts email or username"""
    if request.method == 'POST':
        username_or_email = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        
        if not username_or_email or not password:
            messages.error(request, 'Both fields are required.')
            return render(request, 'login.html', {'form': {'errors': True}})
        
        # Try to find user by email or username
        user = None
        
        if '@' in username_or_email:
            # It's an email - find the user and authenticate with username
            try:
                user_obj = CustomUser.objects.get(email=username_or_email)
                user = authenticate(request, username=user_obj.username, password=password)
            except CustomUser.DoesNotExist:
                user = None
        else:
            # It's a username - authenticate directly
            user = authenticate(request, username=username_or_email, password=password)
        
        if user is not None:
            login(request, user)
            
            # Redirect to next parameter or default to profile
            next_url = request.GET.get('next', 'profile')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid email/username or password.')
            return render(request, 'login.html', {'form': {'errors': True}})
    
    return render(request, 'login.html')


def logout_view(request):
    """Handle user logout"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('index')


@method_decorator(csrf_exempt, name='dispatch')
class AuthGoogle(APIView):
    """Alternative Google authentication handler"""
    def post(self, request, *args, **kwargs):
        try:
            user_data = self.get_google_user_data(request)
        except ValueError:
            return HttpResponse("Invalid Google token", status=403)
        
        email = user_data["email"]
        user, created = CustomUser.objects.get_or_create(
            email=email,
            defaults={
                "username": user_data.get("given_name", email.split('@')[0])[:30],
                "first_name": user_data.get("given_name", ""),
            }
        )
        
        if created:
            user.set_unusable_password()
            user.save()
        
        login(request, user, backend='django.contrib.auth.backends.ModelBackend')
        return redirect('profile')

    @staticmethod
    def get_google_user_data(request: HttpRequest):
        token = request.POST['credential']
        return id_token.verify_oauth2_token(
            token, requests.Request(), settings.GOOGLE_OAUTH_CLIENT_ID
        )


from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

@login_required
@require_POST
def edit_profile(request):
    """Update first name, last name, and username"""
    user = request.user
    first_name = request.POST.get('first_name', '').strip()
    last_name = request.POST.get('last_name', '').strip()
    new_username = request.POST.get('username', '').strip()

    if not new_username:
        messages.error(request, 'Username cannot be empty.')
        return redirect('profile')

    # Check username uniqueness (exclude current user)
    if CustomUser.objects.filter(username=new_username).exclude(pk=user.pk).exists():
        messages.error(request, 'That username is already taken.')
        return redirect('profile')

    user.first_name = first_name
    user.last_name = last_name
    user.username = new_username
    user.save(update_fields=['first_name', 'last_name', 'username'])
    messages.success(request, 'profile_updated')
    return redirect('profile')


@login_required
@require_POST
def edit_email(request):
    """Update email address"""
    user = request.user
    new_email = request.POST.get('email', '').strip()
    password = request.POST.get('password', '').strip()

    if not new_email or not password:
        messages.error(request, 'Email and password are required.')
        return redirect('profile')

    if not user.check_password(password):
        messages.error(request, 'Incorrect password. Email not updated.')
        return redirect('profile')

    if CustomUser.objects.filter(email=new_email).exclude(pk=user.pk).exists():
        messages.error(request, 'That email is already in use.')
        return redirect('profile')

    user.email = new_email
    user.save(update_fields=['email'])
    messages.success(request, 'email_updated')
    return redirect('profile')


@login_required
@require_POST
def change_password(request):
    """Change password"""
    user = request.user
    current_password = request.POST.get('current_password', '').strip()
    new_password = request.POST.get('new_password', '').strip()
    confirm_password = request.POST.get('confirm_password', '').strip()

    if not current_password or not new_password or not confirm_password:
        messages.error(request, 'All password fields are required.')
        return redirect('profile')

    if not user.check_password(current_password):
        messages.error(request, 'Current password is incorrect.')
        return redirect('profile')

    if new_password != confirm_password:
        messages.error(request, 'New passwords do not match.')
        return redirect('profile')

    if len(new_password) < 8:
        messages.error(request, 'Password must be at least 8 characters.')
        return redirect('profile')

    user.set_password(new_password)
    user.save()
    # Keep the user logged in after password change
    from django.contrib.auth import update_session_auth_hash
    update_session_auth_hash(request, user)
    messages.success(request, 'password_changed')
    return redirect('profile')