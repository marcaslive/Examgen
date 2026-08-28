# designer/views/exam_views.py

import json
import logging
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

from ..models import Document, Exam, Question, GradeRule
from ..forms import ExamForm, QuestionForm, GenerateQuestionsForm
from ..services.pdf_service import PDFService
from ..services.question_generator import ExamGenerationManager

logger = logging.getLogger(__name__)


@login_required
def generate_exam_view(request):
    """Page for selecting documents and generating questions."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    documents = Document.objects.filter(uploaded_by=request.user, status='ready')
    return render(request, 'designer/generate_exam.html', {'documents': documents})


# ============================================================
# BATCH GENERATION API (always returns JSON)
# ============================================================

@csrf_exempt
@require_POST
def generate_start_api(request):
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Session expired. Please log in again.'}, status=401)
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        payload = json.loads(request.body.decode('utf-8') or '{}')
    except Exception as e:
        return JsonResponse({"success": False, "error": f"Invalid payload: {e}"}, status=400)

    try:
        request.session['generation_doc_ids'] = payload.get("documents", [])
        status_code, response_data = ExamGenerationManager.start_generation(request, payload)
        return JsonResponse(response_data, status=status_code)
    except Exception as e:
        logger.exception("generate_start_api failed")
        return JsonResponse({'success': False, 'error': f"Init failed: {e}"}, status=500)


@csrf_exempt
@require_POST
def generate_batch_api(request):
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Session expired.'}, status=401)
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        status_code, response_data = ExamGenerationManager.process_next_batch(request)
        return JsonResponse(response_data, status=status_code)
    except Exception as e:
        logger.exception("generate_batch_api failed")
        return JsonResponse({'success': False, 'error': f"Batch error: {e}"}, status=500)


@require_GET
def generate_quota_api(request):
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Session expired.'}, status=401)
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        return JsonResponse({"success": True, **ExamGenerationManager.get_quota_status()})
    except Exception as e:
        logger.exception("generate_quota_api failed")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ============================================================
# EXAM VIEWS
# ============================================================

@login_required
def review_questions_view(request):
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    review_data = ExamGenerationManager.get_review_data(request)
    questions = review_data.get('questions') or []
    doc_ids = review_data.get('document_ids') or request.session.get('generation_doc_ids', [])

    if not questions:
        return redirect('designer:generate_exam')

    request.session['generated_questions'] = questions
    request.session.modified = True

    documents = Document.objects.filter(id__in=doc_ids)
    users = User.objects.filter(is_staff=False, is_active=True)

    return render(request, 'designer/review_questions.html', {
        'questions': questions,
        'documents': documents,
        'users': users,
        'exam_form': ExamForm(),
    })


@csrf_exempt
@require_POST
def save_exam_view(request):
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Session expired.'}, status=401)
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid request body.'}, status=400)

    exam_data = data.get('exam', {})
    questions_data = data.get('questions', [])
    grade_rules_data = data.get('grade_rules', [])
    assigned_user_ids = data.get('assigned_users', [])
    save_as = data.get('save_as', 'draft')

    if not questions_data:
        return JsonResponse({'success': False, 'error': 'No questions to save.'}, status=400)

    try:
        with transaction.atomic():
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
            if exam_data.get('scheduled_date'):
                exam.scheduled_date = exam_data['scheduled_date']
            if exam_data.get('start_time'):
                exam.start_time = exam_data['start_time']
            if exam_data.get('end_time'):
                exam.end_time = exam_data['end_time']

            exam.save()

            doc_ids = request.session.get('generation_doc_ids', [])
            if doc_ids:
                docs = Document.objects.filter(id__in=doc_ids)
                exam.source_documents.set(docs)

            if exam.assignment_type == 'specific' and assigned_user_ids:
                users = User.objects.filter(id__in=assigned_user_ids)
                exam.assigned_users.set(users)

            for idx, q in enumerate(questions_data):
                source_doc = None
                sdi = q.get('source_document_id')
                if sdi:
                    try:
                        source_doc = Document.objects.get(id=sdi)
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

            if grade_rules_data:
                for rule in grade_rules_data:
                    GradeRule.objects.create(
                        exam=exam,
                        grade=rule.get('grade', ''),
                        min_score=int(rule.get('min_score', 0)),
                        max_score=int(rule.get('max_score', 100)),
                    )
            else:
                for grade, min_s, max_s in [
                    ('A', 70, 100), ('B', 60, 69), ('C', 50, 59),
                    ('D', 45, 49), ('E', 40, 44), ('F', 0, 39),
                ]:
                    GradeRule.objects.create(
                        exam=exam, grade=grade, min_score=min_s, max_score=max_s
                    )

            for k in ('review_questions', 'generated_questions', 'generation_doc_ids'):
                request.session.pop(k, None)

        return JsonResponse({
            'success': True,
            'exam_id': str(exam.id),
            'message': f'Exam "{exam.name}" saved as {save_as}.',
            'redirect': f'/exams/{exam.id}/',
        })

    except Exception as e:
        logger.exception("Error saving exam")
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@login_required
def exam_list_view(request):
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')
    return render(request, 'designer/exam_list.html', {'exams': Exam.objects.filter(created_by=request.user)})


@login_required
def exam_detail_view(request, exam_id):
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')
    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    return render(request, 'designer/exam_detail.html', {
        'exam': exam,
        'questions': exam.questions.all(),
        'grade_rules': exam.grade_rules.all(),
        'attempts': exam.attempts.select_related('user').all(),
    })


@login_required
def exam_edit_view(request, exam_id):
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

    return render(request, 'designer/exam_edit.html', {
        'exam': exam,
        'form': form,
        'questions': exam.questions.all(),
        'users': User.objects.filter(is_staff=False, is_active=True),
    })


@csrf_exempt
@require_POST
def exam_publish_view(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
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


@csrf_exempt
@require_POST
def exam_delete_view(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    name = exam.name
    exam.delete()
    return JsonResponse({'success': True, 'message': f'Exam "{name}" deleted.'})


@csrf_exempt
@require_POST
def exam_duplicate_view(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    original = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    with transaction.atomic():
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
        new_exam.source_documents.set(original.source_documents.all())

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
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')
    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    return render(request, 'designer/exam_preview.html', {
        'exam': exam,
        'questions': list(exam.questions.all()),
        'is_preview': True,
    })


@csrf_exempt
@require_POST
def question_edit_api(request, question_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)

    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'}, status=400)

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


@csrf_exempt
@require_POST
def question_delete_api(request, question_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)
    question.delete()
    return JsonResponse({'success': True, 'message': 'Question deleted.'})


@csrf_exempt
@require_POST
def question_add_api(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'}, status=400)

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


@csrf_exempt
@require_POST
def exam_release_results_view(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)
    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    exam.results_released = True
    exam.save()
    return JsonResponse({'success': True, 'message': 'Results released.'})


@csrf_exempt
@require_POST
def exam_assign_users_view(request, exam_id):
    if not request.user.is_authenticated or not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body.decode('utf-8') or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'}, status=400)

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