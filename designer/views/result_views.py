# designer/views/result_views.py

import csv
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.db.models import Q, Avg, Count

from ..models import Exam, ExamAttempt


@login_required
def results_dashboard_view(request):
    """Admin results dashboard."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exams = Exam.objects.filter(created_by=request.user)

    # Filter
    exam_filter = request.GET.get('exam', '')
    status_filter = request.GET.get('status', '')
    search = request.GET.get('search', '')

    attempts = ExamAttempt.objects.filter(
        exam__created_by=request.user
    ).select_related('user', 'exam').exclude(status='in_progress')

    if exam_filter:
        attempts = attempts.filter(exam_id=exam_filter)
    if status_filter == 'passed':
        attempts = attempts.filter(passed=True)
    elif status_filter == 'failed':
        attempts = attempts.filter(passed=False)
    if search:
        attempts = attempts.filter(
            Q(user__username__icontains=search) |
            Q(user__first_name__icontains=search) |
            Q(user__last_name__icontains=search)
        )

    # Sorting
    sort = request.GET.get('sort', '-submitted_at')
    if sort in ['percentage', '-percentage', 'submitted_at', '-submitted_at', 'user__username', '-user__username']:
        attempts = attempts.order_by(sort)

    context = {
        'attempts': attempts,
        'exams': exams,
        'exam_filter': exam_filter,
        'status_filter': status_filter,
        'search': search,
        'sort': sort,
    }
    return render(request, 'designer/results_dashboard.html', context)


@login_required
def export_results_view(request, exam_id):
    """Export results as CSV."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    from ..models import Exam
    from django.shortcuts import get_object_or_404

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    attempts = ExamAttempt.objects.filter(exam=exam).exclude(
        status='in_progress'
    ).select_related('user')

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{exam.name}_results.csv"'

    writer = csv.writer(response)
    writer.writerow([
        'Username', 'First Name', 'Last Name', 'Email',
        'Score', 'Total', 'Percentage', 'Grade', 'Status',
        'Time Spent', 'Submitted At'
    ])

    for a in attempts:
        writer.writerow([
            a.user.username,
            a.user.first_name,
            a.user.last_name,
            a.user.email,
            a.score,
            a.total_questions,
            f"{a.percentage}%",
            a.grade,
            'Passed' if a.passed else 'Failed',
            a.time_spent_display,
            a.submitted_at.strftime('%Y-%m-%d %H:%M') if a.submitted_at else 'N/A',
        ])

    return response


@login_required
def users_list_view(request):
    """List registered users (admin)."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    from django.contrib.auth.models import User
    users = User.objects.filter(is_staff=False).annotate(
        attempt_count=Count('exam_attempts'),
        avg_score=Avg('exam_attempts__percentage'),
    )

    context = {
        'users': users,
    }
    return render(request, 'designer/users_list.html', context)