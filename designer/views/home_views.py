# designer/views/home_views.py

from django.shortcuts import render, redirect


def home(request):
    """
    Landing page.
    - If user is already logged in, redirect to appropriate dashboard.
    - Otherwise show the landing page with Admin / User login options.
    """
    if request.user.is_authenticated:
        # Redirect admins to admin dashboard
        if request.user.is_staff or request.user.is_superuser:
            return redirect('designer:admin_dashboard')
        # Redirect regular users to user dashboard
        else:
            return redirect('designer:user_dashboard')

    return render(request, 'designer/home.html')