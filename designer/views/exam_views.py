# designer/views/exam_views.py

import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.db import transaction

from ..models import Document, Exam, Question, GradeRule
from ..forms import ExamForm, QuestionForm, GenerateQuestionsForm
from ..services.pdf_service import PDFService

# Import the new Manager
from ..services.question_generator import ExamGenerationManager


@login_required
def generate_exam_view(request):
    """Page for selecting documents and generating questions."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    documents = Document.objects.filter(uploaded_by=request.user, status='ready')
    context = {
        'documents': documents,
    }
    return render(request, 'designer/generate_exam.html', context)

# ============================================================
# NEW BATCH GENERATION ENDPOINTS
# ============================================================

@login_required
@require_POST
def generate_start_api(request):
    """AJAX endpoint to initialize generation, check limits, and build batch plan."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid request data."}, status=400)

    # Stash selected documents for the review/save page to use later
    request.session['generation_doc_ids'] = payload.get("documents", [])

    status_code, response_data = ExamGenerationManager.start_generation(request, payload)
    return JsonResponse(response_data, status=status_code)


@login_required
@require_POST
def generate_batch_api(request):
    """AJAX endpoint to process exactly one batch of questions."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    status_code, response_data = ExamGenerationManager.process_next_batch(request)
    return JsonResponse(response_data, status=status_code)


@login_required
@require_GET
def generate_quota_api(request):
    """AJAX endpoint to check remaining AI limits."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    return JsonResponse({"success": True, **ExamGenerationManager.get_quota_status()})


# ============================================================
# EXISTING EXAM VIEWS
# ============================================================

@login_required
def review_questions_view(request):
    """Review generated questions before creating an exam."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    # Look for both the old key and the new key for compatibility
    questions = request.session.get('review_questions') or request.session.get('generated_questions', [])
    doc_ids = request.session.get('generation_doc_ids', [])

    if not questions:
        return redirect('designer:generate_exam')

    documents = Document.objects.filter(id__in=doc_ids)
    users = User.objects.filter(is_staff=False, is_active=True)

    context = {
        'questions': questions,
        'documents': documents,
        'users': users,
        'exam_form': ExamForm(),
    }
    return render(request, 'designer/review_questions.html', context)


@login_required
@require_POST
def save_exam_view(request):
    """Save the exam with generated questions."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    exam_data = data.get('exam', {})
    questions_data = data.get('questions', [])
    grade_rules_data = data.get('grade_rules', [])
    assigned_user_ids = data.get('assigned_users', [])
    save_as = data.get('save_as', 'draft')  # 'draft' or 'published'

    if not questions_data:
        return JsonResponse({'success': False, 'error': 'No questions to save.'})

    try:
        with transaction.atomic():
            # Create exam
            exam = Exam(
                name=exam_data.get('name', 'Untitled Exam'),
                description=exam_data.get('description', ''),
                institution=exam_data.get('institution', ''),
                course=exam_data.get('course', ''),
                department=exam_data.get('department', ''),
                exam_code=exam_data.get('exam_code', ''),
                instructor=exam_data.get('instructor', ''),
                instructions=exam_data.get('instructions', ''),
                answer_mode=exam_data.get('answer_mode', 'exam'),
                result_timing=exam_data.get('result_timing', 'immediate'),
                assignment_type=exam_data.get('assignment_type', 'all'),
                duration_minutes=int(exam_data.get('duration_minutes', 60)),
                max_attempts=int(exam_data.get('max_attempts', 1)),
                pass_mark=int(exam_data.get('pass_mark', 50)),
                randomize_questions=exam_data.get('randomize_questions', False),
                randomize_options=exam_data.get('randomize_options', False),
                pass_message=exam_data.get('pass_message', 'Congratulations! You passed the examination.'),
                fail_message=exam_data.get('fail_message', 'Unfortunately, you did not meet the required pass mark.'),
                status=save_as,
                created_by=request.user,
            )

            # Handle dates
            scheduled_date = exam_data.get('scheduled_date')
            if scheduled_date:
                exam.scheduled_date = scheduled_date
            start_time = exam_data.get('start_time')
            if start_time:
                exam.start_time = start_time
            end_time = exam_data.get('end_time')
            if end_time:
                exam.end_time = end_time

            exam.save()

            # Add source documents
            doc_ids = request.session.get('generation_doc_ids', [])
            if doc_ids:
                docs = Document.objects.filter(id__in=doc_ids)
                exam.source_documents.set(docs)

            # Assign users
            if exam.assignment_type == 'specific' and assigned_user_ids:
                users = User.objects.filter(id__in=assigned_user_ids)
                exam.assigned_users.set(users)

            # Save questions
            for idx, q in enumerate(questions_data):
                source_doc = None
                source_doc_id = q.get('source_document_id')
                if source_doc_id:
                    try:
                        source_doc = Document.objects.get(id=source_doc_id)
                    except Document.DoesNotExist:
                        pass

                Question.objects.create(
                    exam=exam,
                    question_text=q.get('question_text', ''),
                    option_a=q.get('option_a', ''),
                    option_b=q.get('option_b', ''),
                    option_c=q.get('option_c', ''),
                    option_d=q.get('option_d', ''),
                    correct_answer=q.get('correct_answer', 'A'),
                    explanation=q.get('explanation', ''),
                    source_document=source_doc,
                    source_page=q.get('source_page'),
                    order=idx + 1,
                )

            # Save grade rules
            if grade_rules_data:
                for rule in grade_rules_data:
                    GradeRule.objects.create(
                        exam=exam,
                        grade=rule.get('grade', ''),
                        min_score=int(rule.get('min_score', 0)),
                        max_score=int(rule.get('max_score', 100)),
                    )
            else:
                # Default grading
                default_grades = [
                    ('A', 70, 100),
                    ('B', 60, 69),
                    ('C', 50, 59),
                    ('D', 45, 49),
                    ('E', 40, 44),
                    ('F', 0, 39),
                ]
                for grade, min_s, max_s in default_grades:
                    GradeRule.objects.create(
                        exam=exam, grade=grade, min_score=min_s, max_score=max_s
                    )

            # Clear session data
            request.session.pop('review_questions', None)
            request.session.pop('generated_questions', None)
            request.session.pop('generation_doc_ids', None)

        return JsonResponse({
            'success': True,
            'exam_id': str(exam.id),
            'message': f'Exam "{exam.name}" saved as {save_as}.',
            'redirect': f'/exams/{exam.id}/'
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def exam_list_view(request):
    """List all exams (admin)."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exams = Exam.objects.filter(created_by=request.user)
    context = {'exams': exams}
    return render(request, 'designer/exam_list.html', context)


@login_required
def exam_detail_view(request, exam_id):
    """View exam details."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    questions = exam.questions.all()
    grade_rules = exam.grade_rules.all()
    attempts = exam.attempts.select_related('user').all()

    context = {
        'exam': exam,
        'questions': questions,
        'grade_rules': grade_rules,
        'attempts': attempts,
    }
    return render(request, 'designer/exam_detail.html', context)


@login_required
def exam_edit_view(request, exam_id):
    """Edit an exam."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    if request.method == 'POST':
        form = ExamForm(request.POST, instance=exam)
        if form.is_valid():
            form.save()
            return redirect('designer:exam_detail', exam_id=exam.id)
    else:
        form = ExamForm(instance=exam)

    questions = exam.questions.all()
    users = User.objects.filter(is_staff=False, is_active=True)

    context = {
        'exam': exam,
        'form': form,
        'questions': questions,
        'users': users,
    }
    return render(request, 'designer/exam_edit.html', context)


@login_required
@require_POST
def exam_publish_view(request, exam_id):
    """Publish/unpublish an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    if exam.status in ['draft', 'unpublished']:
        if exam.questions.count() == 0:
            return JsonResponse({'success': False, 'error': 'Cannot publish an exam with no questions.'})
        exam.status = 'published'
        exam.save()
        return JsonResponse({'success': True, 'status': 'published', 'message': 'Exam published.'})
    elif exam.status == 'published':
        exam.status = 'unpublished'
        exam.save()
        return JsonResponse({'success': True, 'status': 'unpublished', 'message': 'Exam unpublished.'})

    return JsonResponse({'success': False, 'error': 'Invalid status transition.'})


@login_required
@require_POST
def exam_delete_view(request, exam_id):
    """Delete an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    name = exam.name
    exam.delete()
    return JsonResponse({'success': True, 'message': f'Exam "{name}" deleted.'})


@login_required
@require_POST
def exam_duplicate_view(request, exam_id):
    """Duplicate an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    original = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    with transaction.atomic():
        # Create exam copy
        new_exam = Exam.objects.create(
            name=f"{original.name} (Copy)",
            description=original.description,
            institution=original.institution,
            course=original.course,
            department=original.department,
            exam_code=f"{original.exam_code}-COPY" if original.exam_code else '',
            instructor=original.instructor,
            instructions=original.instructions,
            answer_mode=original.answer_mode,
            result_timing=original.result_timing,
            assignment_type=original.assignment_type,
            duration_minutes=original.duration_minutes,
            max_attempts=original.max_attempts,
            pass_mark=original.pass_mark,
            randomize_questions=original.randomize_questions,
            randomize_options=original.randomize_options,
            pass_message=original.pass_message,
            fail_message=original.fail_message,
            status='draft',
            created_by=request.user,
        )

        # Copy source documents
        new_exam.source_documents.set(original.source_documents.all())

        # Copy questions
        for q in original.questions.all():
            Question.objects.create(
                exam=new_exam,
                question_text=q.question_text,
                option_a=q.option_a,
                option_b=q.option_b,
                option_c=q.option_c,
                option_d=q.option_d,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                source_document=q.source_document,
                source_page=q.source_page,
                order=q.order,
            )

        # Copy grade rules
        for rule in original.grade_rules.all():
            GradeRule.objects.create(
                exam=new_exam,
                grade=rule.grade,
                min_score=rule.min_score,
                max_score=rule.max_score,
            )

    return JsonResponse({
        'success': True,
        'exam_id': str(new_exam.id),
        'message': f'Exam duplicated as "{new_exam.name}".',
    })


@login_required
def exam_preview_view(request, exam_id):
    """Preview an exam as a student would see it."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    questions = list(exam.questions.all())

    context = {
        'exam': exam,
        'questions': questions,
        'is_preview': True,
    }
    return render(request, 'designer/exam_preview.html', context)


@login_required
@require_POST
def question_edit_api(request, question_id):
    """Edit a question (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    question.question_text = data.get('question_text', question.question_text)
    question.option_a = data.get('option_a', question.option_a)
    question.option_b = data.get('option_b', question.option_b)
    question.option_c = data.get('option_c', question.option_c)
    question.option_d = data.get('option_d', question.option_d)
    correct = data.get('correct_answer', question.correct_answer)
    if correct in ['A', 'B', 'C', 'D']:
        question.correct_answer = correct
    question.explanation = data.get('explanation', question.explanation)
    question.save()

    return JsonResponse({'success': True, 'message': 'Question updated.'})


@login_required
@require_POST
def question_delete_api(request, question_id):
    """Delete a question (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)
    question.delete()
    return JsonResponse({'success': True, 'message': 'Question deleted.'})


@login_required
@require_POST
def question_add_api(request, exam_id):
    """Add a question manually (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    max_order = exam.questions.count()

    question = Question.objects.create(
        exam=exam,
        question_text=data.get('question_text', 'New question'),
        option_a=data.get('option_a', 'Option A'),
        option_b=data.get('option_b', 'Option B'),
        option_c=data.get('option_c', 'Option C'),
        option_d=data.get('option_d', 'Option D'),
        correct_answer=data.get('correct_answer', 'A'),
        explanation=data.get('explanation', ''),
        order=max_order + 1,
    )

    return JsonResponse({
        'success': True,
        'question_id': str(question.id),
        'message': 'Question added.',
    })


@login_required
@require_POST
def exam_release_results_view(request, exam_id):
    """Release results for an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    exam.results_released = True
    exam.save()
    return JsonResponse({'success': True, 'message': 'Results released.'})


@login_required
@require_POST
def exam_assign_users_view(request, exam_id):
    """Assign users to an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    assignment_type = data.get('assignment_type', exam.assignment_type)
    user_ids = data.get('user_ids', [])

    exam.assignment_type = assignment_type
    exam.save()

    if assignment_type == 'specific':
        users = User.objects.filter(id__in=user_ids)
        exam.assigned_users.set(users)
    else:
        exam.assigned_users.clear()

    return JsonResponse({'success': True, 'message': 'Assignment updated.'})