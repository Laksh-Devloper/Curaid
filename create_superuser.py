#!/usr/bin/env python3
"""
Script to create a superuser with known credentials
Run this with: python3 create_superuser.py
"""
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'case_companion.settings')
django.setup()

from accounts.models import CustomUser
from django.contrib.auth import authenticate

def create_superuser():
    # Superuser credentials
    email = "superadmin@cura-sphere.com"
    username = "superadmin"
    password = "admin123"
    
    print("=" * 60)
    print("Creating Superuser for Cura Sphere")
    print("=" * 60)
    
    # Check if user exists
    if CustomUser.objects.filter(email=email).exists():
        user = CustomUser.objects.get(email=email)
        print(f"⚠️  User '{username}' already exists. Updating password...")
    elif CustomUser.objects.filter(username=username).exists():
        user = CustomUser.objects.get(username=username)
        print(f"⚠️  User '{username}' already exists. Updating password...")
    else:
        user = CustomUser.objects.create_superuser(
            email=email,
            username=username,
            password=password
        )
        print(f"✅ Created new superuser: {username}")
    
    # Ensure password is set correctly
    user.set_password(password)
    user.is_staff = True
    user.is_superuser = True
    user.is_active = True
    user.save()
    
    print("\n" + "=" * 60)
    print("✅ SUPERUSER CREDENTIALS")
    print("=" * 60)
    print(f"Username: {username}")
    print(f"Email: {email}")
    print(f"Password: {password}")
    print(f"\nIs staff: {user.is_staff}")
    print(f"Is superuser: {user.is_superuser}")
    print(f"Is active: {user.is_active}")
    print("=" * 60)
    
    # Test authentication
    test_auth = authenticate(username=username, password=password)
    if test_auth:
        print("\n✅ Authentication test: SUCCESS")
        print(f"\nYou can now login at:")
        print(f"  - Admin panel: http://127.0.0.1:8000/admin/")
        print(f"  - Main app: http://127.0.0.1:8000/login/")
    else:
        print("\n❌ Authentication test: FAILED")
        print("There may be an issue with the authentication backend.")
    
    print("=" * 60)

if __name__ == "__main__":
    create_superuser()
