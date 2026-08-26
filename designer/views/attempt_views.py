# designer/views/attempt_views.py

import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.db.models import Q

from ..models import Exam, ExamAttempt, UserAnswer, Question


@login_required
def start_exam_view(request, exam_id):
    """Start an exam attempt."""
    exam = get_object_or_404(Exam, id=exam_id, status='published')
    user = request.user

    # Check if user is allowed
    if not _user_can_access_exam(user, exam):
        return render(request, 'designer/exam_denied.html', {
            'message': 'You do not have permission to take this exam.'
        })

    # Check attempt limit
    existing_attempts = ExamAttempt.objects.filter(user=user, exam=exam).count()
    if existing_attempts >= exam.max_attempts:
        return render(request, 'designer/exam_denied.html', {
            'message': f'You have reached the maximum number of attempts ({exam.max_attempts}).'
        })

    # Check for in-progress attempt
    in_progress = ExamAttempt.objects.filter(
        user=user, exam=exam, status='in_progress'
    ).first()

    if in_progress:
        # Check if it's expired
        if in_progress.is_expired:
            _auto_submit_attempt(in_progress)
            # Allow starting a new attempt if limit not reached
            completed = ExamAttempt.objects.filter(
                user=user, exam=exam
            ).exclude(status='in_progress').count()
            if completed >= exam.max_attempts:
                return render(request, 'designer/exam_denied.html', {
                    'message': f'Your previous attempt timed out. Maximum attempts reached.'
                })
        else:
            # Resume existing attempt
            return redirect('designer:take_exam', attempt_id=in_progress.id)

    if request.method == 'POST':
        # Create new attempt
        attempt = ExamAttempt.objects.create(
            user=user,
            exam=exam,
            total_questions=exam.questions.count(),
        )

        # Set up question order
        questions = list(exam.questions.values_list('id', flat=True))
        if exam.randomize_questions:
            random.shuffle(questions)
        attempt.question_order = [str(q) for q in questions]

        # Set up option randomization
        if exam.randomize_options:
            option_orders = {}
            for q_id in questions:
                order = ['A', 'B', 'C', 'D']
                random.shuffle(order)
                option_orders[str(q_id)] = order
            attempt.option_orders = option_orders

        attempt.save()

        # Pre-create UserAnswer records
        for q_id in questions:
            UserAnswer.objects.create(
                attempt=attempt,
                question_id=q_id,
            )

        return redirect('designer:take_exam', attempt_id=attempt.id)

    # Show exam info page
    context = {
        'exam': exam,
        'attempt_count': existing_attempts,
    }
    return render(request, 'designer/exam_start.html', context)


@login_required
def take_exam_view(request, attempt_id):
    """The actual examination interface."""
    attempt = get_object_or_404(ExamAttempt, id=attempt_id, user=request.user)

    if attempt.status != 'in_progress':
        return redirect('designer:exam_result', attempt_id=attempt.id)

    # Check if expired server-side
    if attempt.is_expired:
        _auto_submit_attempt(attempt)
        return redirect('designer:exam_result', attempt_id=attempt.id)

    exam = attempt.exam

    # Get questions in the right order
    question_ids = attempt.question_order
    if not question_ids:
        question_ids = [str(q.id) for q in exam.questions.all()]

    questions_map = {str(q.id): q for q in Question.objects.filter(id__in=question_ids)}
    ordered_questions = []
    for q_id in question_ids:
        if q_id in questions_map:
            q = questions_map[q_id]
            q_data = {
                'id': str(q.id),
                'question_text': q.question_text,
                'order': len(ordered_questions) + 1,
            }

            # Handle option randomization
            option_order = attempt.option_orders.get(str(q.id), ['A', 'B', 'C', 'D'])
            options_map = {
                'A': q.option_a,
                'B': q.option_b,
                'C': q.option_c,
                'D': q.option_d,
            }
            q_data['options'] = []
            for i, orig_letter in enumerate(option_order):
                display_letter = chr(65 + i)  # A, B, C, D
                q_data['options'].append({
                    'letter': display_letter,
                    'text': options_map[orig_letter],
                    'value': display_letter,
                })

            ordered_questions.append(q_data)

    # Get current answers
    answers = {}
    for ua in UserAnswer.objects.filter(attempt=attempt):
        if ua.selected_answer:
            answers[str(ua.question_id)] = ua.selected_answer

    context = {
        'attempt': attempt,
        'exam': exam,
        'questions': ordered_questions,
        'answers_json': json.dumps(answers),
        'time_remaining': attempt.time_remaining_seconds,
        'total_questions': len(ordered_questions),
    }

    # For study mode, don't send correct answers still
    if exam.answer_mode == 'study':
        context['is_study_mode'] = True

    return render(request, 'designer/take_exam.html', context)


@login_required
@require_POST
def save_answer_api(request, attempt_id):
    """Save a single answer (AJAX auto-save)."""
    attempt = get_object_or_404(ExamAttempt, id=attempt_id, user=request.user)

    if attempt.status != 'in_progress':
        return JsonResponse({'success': False, 'error': 'Exam already submitted.'})

    if attempt.is_expired:
        _auto_submit_attempt(attempt)
        return JsonResponse({'success': False, 'error': 'Time expired.', 'expired': True})

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    question_id = data.get('question_id')
    selected = data.get('selected_answer', '')

    if selected not in ['A', 'B', 'C', 'D', '']:
        return JsonResponse({'success': False, 'error': 'Invalid answer.'})

    try:
        ua = UserAnswer.objects.get(attempt=attempt, question_id=question_id)
        ua.selected_answer = selected
        ua.save()
        return JsonResponse({'success': True})
    except UserAnswer.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Question not found in attempt.'})


@login_required
@require_POST
def submit_exam_api(request, attempt_id):
    """Submit the exam (AJAX)."""
    attempt = get_object_or_404(ExamAttempt, id=attempt_id, user=request.user)

    if attempt.status != 'in_progress':
        return JsonResponse({
            'success': False,
            'error': 'Exam already submitted.',
            'redirect': f'/attempt/{attempt.id}/result/'
        })

    _submit_attempt(attempt)

    return JsonResponse({
        'success': True,
        'redirect': f'/attempt/{attempt.id}/result/'
    })


@login_required
def exam_result_view(request, attempt_id):
    """Show exam result."""
    attempt = get_object_or_404(ExamAttempt, id=attempt_id, user=request.user)
    exam = attempt.exam

    # Check if results can be shown
    show_result = False
    if exam.result_timing == 'immediate':
        show_result = True
    elif exam.results_released:
        show_result = True
    elif request.user.is_staff and request.user == exam.created_by:
        show_result = True

    # For study mode, always show answers
    if exam.answer_mode == 'study':
        show_result = True

    answers = []
    if show_result or exam.answer_mode == 'study':
        user_answers = UserAnswer.objects.filter(attempt=attempt).select_related('question')

        for ua in user_answers:
            q = ua.question
            # Reverse map the randomized options
            option_order = attempt.option_orders.get(str(q.id), ['A', 'B', 'C', 'D'])
            options_map = {
                'A': q.option_a,
                'B': q.option_b,
                'C': q.option_c,
                'D': q.option_d,
            }

            displayed_options = []
            correct_display = ''
            selected_display = ''

            for i, orig_letter in enumerate(option_order):
                display_letter = chr(65 + i)
                displayed_options.append({
                    'letter': display_letter,
                    'text': options_map[orig_letter],
                })
                if orig_letter == q.correct_answer:
                    correct_display = display_letter
                if ua.selected_answer == display_letter:
                    # Map back to original
                    pass

            answers.append({
                'question': q,
                'user_answer': ua,
                'options': displayed_options,
                'correct_display': correct_display,
                'is_correct': ua.is_correct,
            })

    context = {
        'attempt': attempt,
        'exam': exam,
        'show_result': show_result,
        'answers': answers,
    }
    return render(request, 'designer/exam_result.html', context)


def _user_can_access_exam(user, exam):
    """Check if a user can access an exam."""
    if user.is_staff:
        return True
    if exam.assignment_type == 'all':
        return True
    if exam.assignment_type == 'self' and user == exam.created_by:
        return True
    if exam.assignment_type == 'specific' and exam.assigned_users.filter(id=user.id).exists():
        return True
    return False


def _submit_attempt(attempt):
    """Submit an attempt and calculate the score."""
    exam = attempt.exam

    # Calculate score
    user_answers = UserAnswer.objects.filter(attempt=attempt).select_related('question')
    correct = 0
    total = user_answers.count()

    for ua in user_answers:
        question = ua.question
        option_order = attempt.option_orders.get(str(question.id), ['A', 'B', 'C', 'D'])

        if ua.selected_answer:
            # Map display letter back to original letter
            display_index = ord(ua.selected_answer) - 65  # 0-based
            if 0 <= display_index < len(option_order):
                original_letter = option_order[display_index]
                if original_letter == question.correct_answer:
                    ua.is_correct = True
                    correct += 1
                else:
                    ua.is_correct = False
            else:
                ua.is_correct = False
        else:
            ua.is_correct = False
        ua.save()

    # Update attempt
    attempt.score = correct
    attempt.total_questions = total
    attempt.percentage = round((correct / total * 100), 2) if total > 0 else 0
    attempt.submitted_at = timezone.now()
    attempt.end_time = timezone.now()
    attempt.status = 'graded'

    # Determine grade
    grade_rules = exam.grade_rules.all()
    attempt.grade = ''
    for rule in grade_rules:
        if rule.min_score <= attempt.percentage <= rule.max_score:
            attempt.grade = rule.grade
            break

    # Pass/fail
    attempt.passed = attempt.percentage >= exam.pass_mark

    attempt.save()


def _auto_submit_attempt(attempt):
    """Auto-submit a timed-out attempt."""
    if attempt.status == 'in_progress':
        attempt.status = 'timed_out'
        attempt.save()
        _submit_attempt(attempt)
        attempt.status = 'timed_out'
        attempt.save()