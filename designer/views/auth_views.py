# designer/views/auth_views.py

import json
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.contrib.auth.models import User
from django.db.models import Count, Avg

from ..forms import UserSignupForm, AdminLoginForm, UserLoginForm
from ..models import Document, Exam, ExamAttempt


def home_view(request):
    """Landing page with Admin and User options."""
    return render(request, 'designer/home.html')


def admin_login_view(request):
    """Admin login page."""
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)

        if user is not None and user.is_staff:
            login(request, user)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect': '/admin-dashboard/'})
            return redirect('designer:admin_dashboard')
        else:
            error = 'Invalid credentials or insufficient permissions.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error})
            return render(request, 'designer/admin_login.html', {'error': error})

    return render(request, 'designer/admin_login.html')


def user_login_view(request):
    """User login page."""
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect': '/user-dashboard/'})
            return redirect('designer:user_dashboard')
        else:
            error = 'Invalid username or password.'
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'error': error})
            return render(request, 'designer/user_login.html', {'error': error})

    return render(request, 'designer/user_login.html')


def user_signup_view(request):
    """User registration page."""
    if request.method == 'POST':
        form = UserSignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect': '/user-dashboard/'})
            return redirect('designer:user_dashboard')
        else:
            errors = {field: [str(e) for e in errs] for field, errs in form.errors.items()}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return JsonResponse({'success': False, 'errors': errors})
            return render(request, 'designer/user_signup.html', {'form': form})

    form = UserSignupForm()
    return render(request, 'designer/user_signup.html', {'form': form})


def logout_view(request):
    """Logout and redirect to home."""
    logout(request)
    return redirect('designer:home')


@login_required
def admin_dashboard_view(request):
    """Admin dashboard with statistics."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    total_documents = Document.objects.filter(uploaded_by=request.user).count()
    total_exams = Exam.objects.filter(created_by=request.user).count()
    published_exams = Exam.objects.filter(created_by=request.user, status='published').count()
    total_users = User.objects.filter(is_staff=False).count()
    total_attempts = ExamAttempt.objects.filter(exam__created_by=request.user).count()
    avg_score = ExamAttempt.objects.filter(
        exam__created_by=request.user, status__in=['submitted', 'graded']
    ).aggregate(avg=Avg('percentage'))['avg'] or 0

    recent_exams = Exam.objects.filter(created_by=request.user)[:5]
    recent_attempts = ExamAttempt.objects.filter(
        exam__created_by=request.user
    ).select_related('user', 'exam')[:10]

    context = {
        'total_documents': total_documents,
        'total_exams': total_exams,
        'published_exams': published_exams,
        'total_users': total_users,
        'total_attempts': total_attempts,
        'avg_score': round(avg_score, 1),
        'recent_exams': recent_exams,
        'recent_attempts': recent_attempts,
    }
    return render(request, 'designer/admin_dashboard.html', context)


@login_required
def user_dashboard_view(request):
    """User dashboard showing available exams."""
    user = request.user

    if user.is_staff:
        # Staff can see all published exams
        available_exams = Exam.objects.filter(status='published')
    else:
        # Regular users see exams assigned to them or to all users
        from django.db.models import Q
        available_exams = Exam.objects.filter(
            Q(status='published') & (
                Q(assignment_type='all') |
                Q(assignment_type='self', created_by=user) |
                Q(assignment_type='specific', assigned_users=user)
            )
        ).distinct()

    # Get attempt info for each exam
    exam_data = []
    for exam in available_exams:
        attempts = ExamAttempt.objects.filter(user=user, exam=exam)
        attempt_count = attempts.count()
        last_attempt = attempts.first()
        can_attempt = attempt_count < exam.max_attempts

        exam_data.append({
            'exam': exam,
            'attempt_count': attempt_count,
            'last_attempt': last_attempt,
            'can_attempt': can_attempt,
        })

    context = {
        'exam_data': exam_data,
    }
    return render(request, 'designer/user_dashboard.html', context)